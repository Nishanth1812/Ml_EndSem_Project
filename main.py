from pathlib import Path
import sys

import streamlit.web.bootstrap as stb


def main() -> None:
    app_path = Path(__file__).resolve().parent / "src" / "myapp.py"
    sys.argv = ["streamlit", "run", str(app_path)]
    stb.run()


if __name__ == "__main__":
    main()
