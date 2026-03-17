import email.parser
import json
import subprocess


def scan_pkgs():
    pkgs = json.loads(subprocess.check_output(["conda", "list", "--json"]))

    for pkg in [a for a in pkgs if a["channel"] == "pypi"]:
        print(pkg["name"])

        parser = email.parser.BytesParser()
        m = parser.parsebytes(
            subprocess.check_output(["pip", "show", "-f", pkg["name"]])
        )
        prefix = m.get("Location")
        print(m)
        files = [x.strip() for x in m.get("Files").splitlines()]


scan_pkgs()
