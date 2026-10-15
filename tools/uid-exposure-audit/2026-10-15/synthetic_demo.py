#!/usr/bin/env python3
"""Deterministic invented-only examples and regression checks; stdlib only."""

import copy
import csv
import io
import json
import os
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

from overlap_audit import audit_manifest, main


def run():
    results = []
    with tempfile.TemporaryDirectory(prefix="invented-knee-overlap-") as temporary:
        folder = Path(temporary)

        def csv_file(name, rows):
            path = folder / name
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["StudyInstanceUID"])
                writer.writerows([[value] for value in rows])
            return name

        validation = csv_file("validation.csv", ["SYNTH_VALID_001", "SYNTH_VALID_002"])
        training = csv_file("training.csv", ["SYNTH_TRAIN_001", "SYNTH_TRAIN_002"])
        empty = csv_file("explicitly_empty.csv", [])
        overlap = csv_file("overlap.csv", ["SYNTH_VALID_002", "SYNTH_TRAIN_003"])
        duplicate = csv_file("duplicate.csv", ["SYNTH_VALID_001", "SYNTH_VALID_001"])
        missing = csv_file("missing.csv", ["SYNTH_VALID_001", ""])
        whitespace = csv_file("whitespace.csv", [" SYNTH_VALID_001"])
        pool = lambda filename: {"complete": True, "csv": filename}
        original = {"schema_version": 1, "uid_column": "StudyInstanceUID",
                    "validation_csv": validation, "arms_complete": True,
                    "arms": [{"training": pool(training), "selection": pool(empty)},
                             {"training": pool(training), "selection": pool(empty)}],
                    "ensemble_training": pool(empty), "ensemble_selection": pool(empty)}

        def check(label, config, passes=False, expected_overlap=None):
            manifest = folder / "manifest.json"
            manifest.write_text(json.dumps(config), encoding="utf-8")
            report = audit_manifest(manifest)
            assert (report["uid_exposure_gate"] == "PASS_DECLARED_MANIFESTS") is passes
            assert report["full_ensemble_oof_claim"] != "OOF"
            if not passes:
                assert report["full_ensemble_oof_claim"] == "REFUSED"
            serialized = json.dumps(report)
            for forbidden in ("SYNTH_", temporary, "validation.csv", "training.csv"):
                assert forbidden not in serialized, "Report must never expose identifiers or paths"
            if expected_overlap is not None:
                assert report["arms"][1]["training"]["overlap_uids"] == expected_overlap
            results.append({"case": label, "gate": report["uid_exposure_gate"],
                            "oof_claim": report["full_ensemble_oof_claim"], "passed": True})
            return report

        check("disjoint_declared_pools", original, passes=True)
        config = copy.deepcopy(original)
        config["arms"][1]["training"] = pool(overlap)
        check("one_contributing_arm_overlap", config, expected_overlap=1)
        config = copy.deepcopy(original)
        config["arms"][0]["selection"] = pool(overlap)
        check("checkpoint_selection_exposure", config)
        for label, filename in [("missing_validation_uid", missing),
                                ("duplicate_validation_uid", duplicate),
                                ("uid_whitespace_is_not_normalized", whitespace)]:
            config = copy.deepcopy(original)
            config["validation_csv"] = filename
            check(label, config)
        config = copy.deepcopy(original)
        config["arms"][0]["training"] = {"complete": False}
        check("unknown_training_pool", config)
        config = copy.deepcopy(original)
        config["arms"][0]["selection"] = {"complete": False, "csv": empty}
        check("empty_but_incomplete_selection_pool", config)
        config = copy.deepcopy(original)
        config["arms_complete"] = False
        check("omitted_arm_coverage_unknown", config)
        for stage in ("training", "selection"):
            config = copy.deepcopy(original)
            config["ensemble_" + stage] = pool(overlap)
            check("ensemble_" + stage + "_exposure", config)
        config = copy.deepcopy(original)
        config["arms"][1]["training"] = pool(duplicate)
        check("duplicate_training_uid", config)
        config = copy.deepcopy(original)
        config["validation_csv"] = empty
        check("empty_validation", config)
        config = copy.deepcopy(original)
        config["validation_csv"] = csv_file("zeros.csv", ["001", "002"])
        for arm in config["arms"]:
            arm["training"] = pool(csv_file("numeric_looking.csv", ["1", "2"]))
        check("leading_zero_ids_remain_distinct_strings", config, passes=True)
        for label, content in [("duplicate_header", "StudyInstanceUID,StudyInstanceUID\nSYNTH_A,SYNTH_B\n"),
                               ("malformed_csv", 'StudyInstanceUID\n"SYNTH_UNCLOSED\n')]:
            folder.joinpath("malformed.csv").write_text(content, encoding="utf-8")
            config = copy.deepcopy(original)
            config["validation_csv"] = "malformed.csv"
            check(label, config)
        folder.joinpath("bad_manifest.json").write_text('{"schema_version":1,"schema_version":1}')
        assert audit_manifest(folder / "bad_manifest.json")["full_ensemble_oof_claim"] == "REFUSED"
        results.append({"case": "duplicate_json_keys", "passed": True})
        captured = io.StringIO()
        with redirect_stdout(captured):
            status = main([str(folder / "SYNTH_PRIVATE_IDENTIFIER.json")])
        assert status == 2 and "SYNTH_PRIVATE_IDENTIFIER" not in captured.getvalue()
        results.append({"case": "unreadable_path_is_not_echoed", "passed": True})
        if hasattr(os, "mkfifo"):
            fifo = folder / "invented_fifo"
            os.mkfifo(fifo)
            assert audit_manifest(fifo)["full_ensemble_oof_claim"] == "REFUSED"
            results.append({"case": "fifo_manifest_refused_without_blocking", "passed": True})
            config = copy.deepcopy(original)
            config["validation_csv"] = fifo.name
            check("fifo_csv_refused_without_blocking", config)
    return {"input_kind": "DETERMINISTIC_INVENTED_UIDS_ONLY", "checks": len(results),
            "passed": len(results), "cases": results}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
