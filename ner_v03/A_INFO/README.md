# NER App Run Guide

This project has two parts:

- A Python FastAPI backend in `backends/`
- A Flutter desktop app in the project root

Run the backend first, then run the Flutter app. The Flutter app sends requests to:

```text
http://127.0.0.1:8000/analyze
```

## Requirements

Install these before running the project:

- Python 3.10 or newer
- Flutter SDK
- Git

For Windows desktop:

- Visual Studio 2022 with the "Desktop development with C++" workload

For macOS desktop:

- Xcode
- CocoaPods, if Flutter asks for it during `flutter doctor`

Check your Flutter setup:

```bash
flutter doctor
```

Fix any required desktop setup issues before continuing.

## Windows

Open PowerShell.

### 1. Go to the project folder

Replace `C:\path\to\ner_v03` with the real folder location on your computer.

```powershell
cd C:\path\to\ner_v03
```

### 2. Create and activate a Python virtual environment

Create the virtual environment only once. If `.venv` already exists, just activate it.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

When the virtual environment is active, PowerShell shows `(.venv)` at the start of the line. That is expected. VS Code may also activate `.venv` automatically when you open a new terminal.

### 3. Install backend dependencies

Do this the first time you set up the project, or when `requirements.txt` changes.

```powershell
cd backends
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the dependencies are already installed in the active virtual environment, you can skip this step. Running it again is usually safe because `pip` checks what is already installed, but it may still take time.

If you see `No module named uvicorn`, run this while `(.venv)` is active:

```powershell
cd C:\path\to\ner_v03\backends
python -m pip install -r requirements.txt
```

### 4. Start the backend

```powershell
cd C:\path\to\ner_v03\backends
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Keep this PowerShell window open. The backend is running when you see Uvicorn listening on `http://127.0.0.1:8000`.

Optional backend check:

Open this in a browser:

```text
http://127.0.0.1:8000/health
```

You should see:

```json
{"status":"ok"}
```

### 5. Run the Flutter app

Open a second PowerShell window.

```powershell
cd C:\path\to\ner_v03
flutter config --enable-windows-desktop
flutter pub get
flutter run -d windows
```

The app window should open. Paste text into the input box and click **Analyze**.

## macOS

Open Terminal.

### 1. Go to the project folder

Replace `/path/to/ner_v03` with the real folder location on your computer.

```bash
cd /path/to/ner_v03
```

### 2. Create and activate a Python virtual environment

Create the virtual environment only once. If `.venv` already exists, just activate it.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

When the virtual environment is active, Terminal shows `(.venv)` at the start of the line. That is expected.

### 3. Install backend dependencies

Do this the first time you set up the project, or when `requirements.txt` changes.

```bash
cd backends
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

If the dependencies are already installed in the active virtual environment, you can skip this step. Running it again is usually safe because `pip` checks what is already installed, but it may still take time.

If you see `No module named uvicorn`, run this while `(.venv)` is active:

```bash
cd /path/to/ner_v03/backends
python3 -m pip install -r requirements.txt
```

### 4. Start the backend

```bash
cd /path/to/ner_v03/backends
python3 -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Keep this Terminal window open. The backend is running when you see Uvicorn listening on `http://127.0.0.1:8000`.

Optional backend check:

Open this in a browser:

```text
http://127.0.0.1:8000/health
```

You should see:

```json
{"status":"ok"}
```

### 5. Run the Flutter app

Open a second Terminal window.

```bash
cd /path/to/ner_v03
flutter config --enable-macos-desktop
flutter pub get
flutter run -d macos
```

The app window should open. Paste text into the input box and click **Analyze**.

## Example Text To Test

```text
Tim Cook is the CEO of Apple in California. Sundar Pichai leads Google in Mountain View.
```

Expected output:

- Tim Cook, Apple, California
- Sundar Pichai, Google, Mountain View

## Build A Release Version

Windows:

```powershell
cd C:\path\to\ner_v03
flutter build windows
```

macOS:

```bash
cd /path/to/ner_v03
flutter build macos
```

The backend still needs to be running separately unless the project is changed later to package the backend with the app.

## Troubleshooting

If the Flutter app says it could not analyze text:

- Make sure the backend terminal is still running.
- Open `http://127.0.0.1:8000/health` and check that it returns `{"status":"ok"}`.
- Make sure port `8000` is not being used by another program.

If Python cannot find a package:

From the project root folder, run:

```bash
python -m pip install -r backends/requirements.txt
```

On macOS, use `python3` instead of `python` if needed:

```bash
python3 -m pip install -r backends/requirements.txt
```

If Flutter cannot find a desktop device:

```bash
flutter doctor
```

Then enable the correct desktop target:

```bash
flutter config --enable-windows-desktop
flutter config --enable-macos-desktop
```

Run only the command for your operating system if you are not setting up both.
