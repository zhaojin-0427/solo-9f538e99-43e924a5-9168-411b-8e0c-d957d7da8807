"""叠压无解复核（失败现场留存 + 凭号复算）集成测试。"""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import create_app
from app.service import solve_snapshot


@pytest.fixture()
def client(tmp_path):
    app = create_app(db_path=str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def _make_site(client, name="复核遗址"):
    r = client.post("/sites", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["record"]


def _add_stratum(client, site_id, trench, layer_no, intervals):
    r = client.post(
        f"/sites/{site_id}/strata",
        json={
            "trench": trench,
            "layer_no": layer_no,
            "intervals": [
                {"year_from": lo, "year_to": hi} for lo, hi in intervals
            ],
        },
    )
    assert r.status_code == 201, r.text
    return r.json()["record"]


def _reject_superposition(client, site_id, lower_id, upper_id):
    r = client.post(
        f"/sites/{site_id}/superpositions",
        json={"lower_id": lower_id, "upper_id": upper_id},
    )
    assert r.status_code == 409, r.text
    return r.json()


def test_rejected_superposition_keeps_409_fields_and_returns_review_id(client):
    site = _make_site(client)
    sid = site["id"]
    low = _add_stratum(client, sid, "T1", "L1", [(100, 100)])
    high = _add_stratum(client, sid, "T2", "L1", [(50, 90)])

    body = _reject_superposition(client, sid, low["id"], high["id"])

    # 原 409 字段一个不少
    assert body["detail"]
    assert body["contradiction"]["reason"] == "ORDER_CONFLICT"
    assert body["previous_solution"]["feasible"] is True
    # 新增：当次复核号与被拒上下层 ID
    assert isinstance(body["review_id"], int)
    assert body["rejected_lower_stratum_id"] == low["id"]
    assert body["rejected_upper_stratum_id"] == high["id"]
    assert "review_persistence_error" not in body


def test_review_query_contains_reason_ids_and_marked_snapshot(client):
    site = _make_site(client)
    sid = site["id"]
    a = _add_stratum(client, sid, "T1", "L1", [(1, 100)])
    b = _add_stratum(client, sid, "T1", "L2", [(1, 100)])
    c = _add_stratum(client, sid, "T2", "L1", [(1, 100)])
    # 已登记关系：叠压 a->b、对接 a-c
    sup_ab = client.post(
        f"/sites/{sid}/superpositions",
        json={"lower_id": a["id"], "upper_id": b["id"]},
    ).json()["record"]
    corr_ac = client.post(
        f"/sites/{sid}/correlations",
        json={"stratum_a_id": a["id"], "stratum_b_id": c["id"]},
    ).json()["record"]

    # 再叠压 b->c：组{a,c} -> b -> 组{a,c}，成环被拒
    body = _reject_superposition(client, sid, b["id"], c["id"])
    review_id = body["review_id"]

    r = client.get(f"/reviews/{review_id}")
    assert r.status_code == 200, r.text
    review = r.json()

    # 无解原因与被拒上下层 ID
    assert review["reason"] == "GROUP_CYCLE"
    assert review["message"]
    assert review["rejected_superposition"] == {
        "lower_stratum_id": b["id"],
        "upper_stratum_id": c["id"],
    }

    snapshot = review["snapshot"]
    # 原始层位年代段全部入快照
    assert {s["id"] for s in snapshot["strata"]} == {a["id"], b["id"], c["id"]}
    assert snapshot["strata"][0]["intervals"] == [{"year_from": 1, "year_to": 100}]

    # 被拒关系与当时已登记关系分别标明
    rejected = [s for s in snapshot["superpositions"] if s["status"] == "rejected"]
    registered = [s for s in snapshot["superpositions"] if s["status"] == "registered"]
    assert len(rejected) == 1
    assert rejected[0] == {
        "id": 0,  # 被拒候选从未入库，保留 id 0
        "lower_id": b["id"],
        "upper_id": c["id"],
        "status": "rejected",
    }
    assert snapshot["rejected_superposition_id"] == 0
    assert registered == [
        {
            "id": sup_ab["id"],
            "lower_id": a["id"],
            "upper_id": b["id"],
            "status": "registered",
        }
    ]
    assert snapshot["correlations"] == [
        {
            "id": corr_ac["id"],
            "stratum_a_id": a["id"],
            "stratum_b_id": c["id"],
            "status": "registered",
        }
    ]

    # 凭返回记录现场复算：同一种无解
    recomputed = review["recomputed"]
    assert recomputed["feasible"] is False
    assert recomputed["contradiction"]["reason"] == "GROUP_CYCLE"


def test_snapshot_alone_reproduces_same_infeasibility_offline(client):
    """仅用查询返回的快照 JSON（不访问数据库）即可复现同一种无解。"""
    site = _make_site(client)
    sid = site["id"]
    low = _add_stratum(client, sid, "T1", "L1", [(100, 100)])
    high = _add_stratum(client, sid, "T2", "L1", [(50, 90)])
    body = _reject_superposition(client, sid, low["id"], high["id"])

    review = client.get(f"/reviews/{body['review_id']}").json()
    # 模拟交班复核员离线环境：只拿到返回记录，直接喂给纯函数求解器
    offline_solution = solve_snapshot(review["snapshot"])
    assert offline_solution.feasible is False
    assert offline_solution.contradiction.reason == review["reason"] == "ORDER_CONFLICT"
    witness_ids = {w.record_id for w in offline_solution.contradiction.witness}
    assert {low["id"], high["id"], 0}.issubset(witness_ids)


def test_review_survives_undo_success_and_restart(tmp_path):
    """事后撤销冲突对接、同一叠压成功入库、服务重启，首次复核仍复现同种无解。"""
    db_path = str(tmp_path / "persist.db")
    with TestClient(create_app(db_path=db_path)) as c1:
        site = _make_site(c1)
        sid = site["id"]
        a = _add_stratum(c1, sid, "T1", "L1", [(1, 100)])
        b = _add_stratum(c1, sid, "T1", "L2", [(1, 100)])
        cc = _add_stratum(c1, sid, "T2", "L1", [(1, 100)])
        c1.post(
            f"/sites/{sid}/superpositions",
            json={"lower_id": a["id"], "upper_id": b["id"]},
        )
        corr = c1.post(
            f"/sites/{sid}/correlations",
            json={"stratum_a_id": a["id"], "stratum_b_id": cc["id"]},
        ).json()["record"]

        rejected = _reject_superposition(c1, sid, b["id"], cc["id"])
        review_id = rejected["review_id"]
        base_version = rejected["previous_solution"]["version"]

        first = c1.get(f"/reviews/{review_id}").json()
        assert first["recomputed"]["contradiction"]["reason"] == "GROUP_CYCLE"

        # 交班：复核员只拿到回执；冲突的对接在此期间被撤掉
        d = c1.delete(f"/correlations/{corr['id']}")
        assert d.status_code == 200
        # 撤销后同一叠压可以成功入库
        ok = c1.post(
            f"/sites/{sid}/superpositions",
            json={"lower_id": b["id"], "upper_id": cc["id"]},
        )
        assert ok.status_code == 201, ok.text

        again = c1.get(f"/reviews/{review_id}").json()
        # 库内版本已前进，但回执记住的 base_version 不变
        assert again["base_version"] == base_version
        assert again["current_version"] > base_version
        # 仅凭历史返回记录，仍是同一种无解
        assert again["recomputed"]["feasible"] is False
        assert again["recomputed"]["contradiction"]["reason"] == "GROUP_CYCLE"
        # 快照里被拒候选依旧在，且不含事后才成功的那条库内新关系语义
        statuses = {s["status"] for s in again["snapshot"]["superpositions"]}
        assert statuses == {"registered", "rejected"}

    # 服务重启：回执持久化在 SQLite 中，复算能力不依赖进程内存
    with TestClient(create_app(db_path=db_path)) as c2:
        after_restart = c2.get(f"/reviews/{review_id}").json()
        assert after_restart["recomputed"]["feasible"] is False
        assert (
            after_restart["recomputed"]["contradiction"]["reason"]
            == "GROUP_CYCLE"
        )
        assert after_restart["snapshot"]["rejected_superposition_id"] == 0


def test_rejection_changes_nothing_in_database(client):
    """无解登记不改变库内关系、version 和年代结论。"""
    site = _make_site(client)
    sid = site["id"]
    low = _add_stratum(client, sid, "T1", "L1", [(100, 100)])
    high = _add_stratum(client, sid, "T2", "L1", [(50, 90)])

    before = client.get(f"/sites/{sid}/solution").json()
    body = _reject_superposition(client, sid, low["id"], high["id"])
    after = client.get(f"/sites/{sid}/solution").json()

    # version 未因被拒或回执登记而 +1
    assert after["version"] == before["version"] == body["previous_solution"]["version"]
    # 库内没有被拒的叠压，结论保持可行且完全一致
    assert client.get(f"/sites/{sid}/superpositions").json() == []
    assert after["feasible"] is True
    assert after["groups"] == before["groups"]

    # 回执的 base_version 与库内版本一致
    review = client.get(f"/reviews/{body['review_id']}").json()
    assert review["base_version"] == before["version"]


def test_persistence_failure_explains_reason_and_gives_no_review_id(client, monkeypatch):
    site = _make_site(client)
    sid = site["id"]
    low = _add_stratum(client, sid, "T1", "L1", [(100, 100)])
    high = _add_stratum(client, sid, "T2", "L1", [(50, 90)])

    def boom(*args, **kwargs):
        raise sqlite3.Error("simulated disk failure")

    monkeypatch.setattr(db, "create_rejection_review", boom)

    r = client.post(
        f"/sites/{sid}/superpositions",
        json={"lower_id": low["id"], "upper_id": high["id"]},
    )
    assert r.status_code == 409
    body = r.json()
    # 原 409 内容仍在
    assert body["contradiction"]["reason"] == "ORDER_CONFLICT"
    assert body["previous_solution"]["feasible"] is True
    # 留存失败：说明原因且不给复核号
    assert body["review_id"] is None
    assert "review_persistence_error" in body
    assert "simulated disk failure" in body["review_persistence_error"]
    # 没有留下任何回执
    assert client.get(f"/sites/{sid}/reviews").json() == []
    # 业务数据原样
    assert client.get(f"/sites/{sid}/superpositions").json() == []


def test_review_listing_and_404(client):
    site = _make_site(client)
    sid = site["id"]
    s1 = _add_stratum(client, sid, "T1", "L1", [(100, 100)])
    s2 = _add_stratum(client, sid, "T2", "L1", [(50, 90)])
    s3 = _add_stratum(client, sid, "T3", "L1", [(40, 40)])

    r1 = _reject_superposition(client, sid, s1["id"], s2["id"])
    r2 = _reject_superposition(client, sid, s1["id"], s3["id"])
    assert r1["review_id"] != r2["review_id"]

    listing = client.get(f"/sites/{sid}/reviews").json()
    assert [item["review_id"] for item in listing] == [
        r1["review_id"],
        r2["review_id"],
    ]
    assert listing[0]["reason"] == "ORDER_CONFLICT"
    assert listing[0]["rejected_superposition"]["lower_stratum_id"] == s1["id"]

    assert client.get("/reviews/999999").status_code == 404
    # 不存在的遗址下列表也是 404
    assert client.get("/sites/999999/reviews").status_code == 404


def test_each_rejection_records_the_state_at_that_moment(client):
    """两次被拒各自快照独立：后一次的已登记关系变化不污染前一次现场。"""
    site = _make_site(client)
    sid = site["id"]
    a = _add_stratum(client, sid, "T1", "L1", [(50, 50)])
    b = _add_stratum(client, sid, "T1", "L2", [(100, 100)])
    c = _add_stratum(client, sid, "T2", "L1", [(1, 100)])

    first = _reject_superposition(client, sid, b["id"], a["id"])  # 100 在下、50 在上，次序冲突
    # 登记一条新叠压后再次被拒：a->c 已在，再提 c->a 即成环
    client.post(
        f"/sites/{sid}/superpositions",
        json={"lower_id": a["id"], "upper_id": c["id"]},
    )
    second = _reject_superposition(client, sid, c["id"], a["id"])

    snap1 = client.get(f"/reviews/{first['review_id']}").json()["snapshot"]
    snap2 = client.get(f"/reviews/{second['review_id']}").json()["snapshot"]

    reg1 = [s for s in snap1["superpositions"] if s["status"] == "registered"]
    reg2 = [s for s in snap2["superpositions"] if s["status"] == "registered"]
    assert reg1 == []  # 首次被拒时库内还没有任何叠压
    assert len(reg2) == 1 and reg2[0]["lower_id"] == a["id"]
    # 两份快照各自都能独立复算无解
    assert solve_snapshot(snap1).feasible is False
    assert solve_snapshot(snap2).feasible is False
