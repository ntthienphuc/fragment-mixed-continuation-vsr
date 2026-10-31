"""Resumable HTTPS transfers and traversal-safe extraction."""

import base64
import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from .common import digest


from .http_transfer import download


def extract(archive, destination):
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Extraction destination must be empty')
    destination.mkdir(parents=True, exist_ok=True)

    def safe(name):
        target = (destination / name).resolve()
        if target != destination and destination not in target.parents:
            raise ValueError('Archive member escapes destination')
        return target

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                safe(member.filename)
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Archive symlinks are not accepted')
            handle.extractall(destination)
    else:
        with tarfile.open(archive) as handle:
            members = handle.getmembers()
            for member in members:
                safe(member.name)
                if not member.isfile() and not member.isdir():
                    raise ValueError('Archive links and special files are not accepted')
            handle.extractall(destination, members=members, filter='data')
