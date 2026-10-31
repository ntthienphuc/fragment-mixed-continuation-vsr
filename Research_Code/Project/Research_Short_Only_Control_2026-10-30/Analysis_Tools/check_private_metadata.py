"""Test the private-host binding handoff, including rejection of altered metadata."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import restore_inputs


def archive(path, payload):
    manifest = {}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, content in payload.items():
            bundle.writestr(name, content)
            manifest[name] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        bundle.writestr("Manifest.json", json.dumps(manifest))
    return {"local": str(path), "bytes": path.stat().st_size,
            "sha256": restore_inputs.sha(path), "verified_members": len(manifest)}


class PrivateMetadataChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "Assessment").mkdir()
        for name in ("Plan.json", "Code_Lock.json"):
            (self.root / "Assessment" / name).write_bytes(b'{"test":"unchanged"}')
        self.training = self.root / "Training.zip"
        archive(self.training, {"Training_Verification.json": b'{"synthetic_fixture":true}'})
        self.payload = {
            "Assessment/Plan.json": b'{"test":"unchanged"}',
            "Assessment/Code_Lock.json": b'{"test":"unchanged"}',
            "Private_Assessment/Training_Evidence/Training_Verification.json": b'{"synthetic_fixture":true}',
            "Private_Assessment/Terminal.json": json.dumps({"status": "all_24_shards_verified",
                                                             "finished": ["synthetic_fixture"]}).encode(),
            "Assessment/Binding_Private/Binding.json": b'{"binding":"fixture"}',
            "Assessment/Binding_Private/model_registry.json": b'{"registry":"fixture"}',
        }
        self.metadata = self.root / "Private.zip"
        self.terminal = {"execution_host": "private_windows", "finished": ["synthetic_fixture"]}
        self.refresh()

    def refresh(self):
        receipt = archive(self.metadata, self.payload)
        path = self.root / "Private_Metadata_Backup.json"
        path.write_text(json.dumps(receipt))
        self.terminal["private_metadata_receipt_sha256"] = restore_inputs.sha(path)

    def tearDown(self):
        self.temporary.cleanup()

    def test_exact_binding_bytes_restored(self):
        with patch.object(restore_inputs, "PHASE", self.root):
            result = restore_inputs.binding_metadata(self.terminal, self.training)
        self.assertEqual(result["Binding.json"], self.payload["Assessment/Binding_Private/Binding.json"])

    def test_changed_training_proof_rejected(self):
        self.payload["Private_Assessment/Training_Evidence/Training_Verification.json"] = b"changed"
        self.refresh()
        with patch.object(restore_inputs, "PHASE", self.root), self.assertRaises(AssertionError):
            restore_inputs.binding_metadata(self.terminal, self.training)

    def test_changed_plan_rejected(self):
        self.payload["Assessment/Plan.json"] = b"changed"
        self.refresh()
        with patch.object(restore_inputs, "PHASE", self.root), self.assertRaises(AssertionError):
            restore_inputs.binding_metadata(self.terminal, self.training)

    def test_changed_archive_rejected(self):
        with self.metadata.open("ab") as stream:
            stream.write(b"corrupted")
        with patch.object(restore_inputs, "PHASE", self.root), self.assertRaises(AssertionError):
            restore_inputs.binding_metadata(self.terminal, self.training)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(PrivateMetadataChecks))
    proof = {"tests": result.testsRun, "passed": result.wasSuccessful(), "new_recognition_result": False}
    Path(__file__).with_name("Private_Metadata_Check.json").write_text(json.dumps(proof, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
