# stt-pipeline: fine-tune a Hebrew Whisper model on your own labeled audio, offline, on the CPU

## How to run (the short version)

```
git clone <the address of this repository> stt
cd stt
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --no-index --find-links wheels -r requirements.lock     # offline; see "Install" below
Copy-Item configs\work_cpu.yaml configs\my_work_cpu.yaml            # then edit 4 lines in the copy
.\.venv\Scripts\python.exe tools\verify_clone.py --config configs\my_work_cpu.yaml
.\tools\run_work.ps1 -Step full -Config configs\my_work_cpu.yaml    # the whole run, no questions
# when it ends it prints the path of the report: open that report.html in a browser
```

Everything else is explained below. Run every command from the project folder (the folder that holds `src\`, `tools\`, `configs\`).

## What it does

It takes labeled audio (clips of at most 30 seconds with their text), splits it by lesson into Train / Val / Test, trains the decoder of the `ivrit-ai/whisper-large-v3-turbo` model (the encoder is frozen, and its output is cached so that training is about 8 times faster on the CPU), scores the model before and after training, and writes ONE report page that shows whether the training converged, was long enough, or overfit. The Test split is never opened unless you ask for it with `-WithTest`. It needs no internet and no administrator rights.

## What you need

| Item | Value |
|---|---|
| Python | 3.12 (tested at home; any version works if the wheel bundle was made for it) |
| System | Windows 11 (tested). Linux: `tools/run_work.sh` is written but UNTESTED |
| RAM | 16 GB installed; the training peaks at 6.4 to 6.7 GB, so close other programs |
| Disk | at least 30 GB free |
| GPU | not needed (CPU only) |
| Model | folder `models\whisper-large-v3-turbo-ivrit` (11 files, 3.24 GB) |
| Data | a folder with `manifest.csv` and the audio files (see `docs\DATA_ONBOARDING.md`) |

## Install

1. Python 3.12 for your user. Put the project at a SHORT path (for example `C:\stt`).
2. `python -m venv .venv`, then `.\.venv\Scripts\Activate.ps1`.
3. Install the pinned packages (`requirements.lock` lists exact versions):
   - with internet: `pip install -r requirements.lock`
   - closed machine (no internet): the packages come from a folder `wheels\` (it is NOT in git, it is about 230 MB). Make it once on a machine WITH internet and the same Python version: `python tools\fetch_wheels.py --requirements requirements.txt --out wheels --lock requirements.lock --python-version 3.12 --platform win_amd64`; copy the `wheels\` folder to the closed machine; then `pip install --no-index --find-links wheels -r requirements.lock`.
4. Check: `.\.venv\Scripts\python.exe tools\verify_clone.py` (it lists what is missing).

## The model and the data

- Model: on a machine with internet, `python tools\fetch_model.py --repo ivrit-ai/whisper-large-v3-turbo --out models\whisper-large-v3-turbo-ivrit`; copy the folder `models\whisper-large-v3-turbo-ivrit` to the same place in the clone.
- Data: any folder, for example `D:\mydata`, with `manifest.csv` (UTF-8) and the audio. The manifest needs a column for the audio file name, a column for the text, and a column for the group (the lesson or recording id; clips of one group never go to two parts). Clips must be at most 30 s.
- Edit `configs\my_work_cpu.yaml`: only 4 lines (the data folder and the three column names; the top of the file says which lines).

## The commands

| Command | What it does |
|---|---|
| `.\tools\run_work.ps1 -Step probe` | what the machine has (RAM, cores, disk, model folder) |
| `.\tools\run_work.ps1 -Step full -Config configs\my_work_cpu.yaml` | the whole run: probe, check, prepare, baseline, train, evaluate, compare, report |
| same with `-Epochs 3` | 3 epochs instead of 2 |
| same with `-WithTest` | also the ONE-TIME Test scoring (only on the final chosen run) |
| the same `full` command again | resumes after a stop or a crash |
| `python -m stt report --experiment exp_NNNN --config configs\my_work_cpu.yaml` | rebuild the report |

The report is `data\work_cpu\experiments\exp_NNNN\reports\report.html` (the script prints the exact path). Open it in a browser; it needs no network. How to read it, what each number means and what to do when something fails: `docs\WORK_RUNBOOK.md` and `docs\USER_GUIDE.md`. The parts of the system: `docs\COMPONENT_GUIDE.md`.

## Folders

`src\stt\` the code. `tools\` the scripts. `configs\` the settings. `docs\` the documents. `tests\` the tests (`python -m unittest discover -s tests`, about 10 minutes). `models\`, `data\`, `wheels\`, `experiments\`, `registry\` are NOT in git.

---

# stt-pipeline: כיוון עדין של מודל Whisper בעברית על האודיו המתויג שלכם, בלי אינטרנט, על המעבד

## הפעלה מהירה

```
git clone <כתובת המאגר> stt
cd stt
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --no-index --find-links wheels -r requirements.lock     # בלי אינטרנט, ראו "התקנה"
Copy-Item configs\work_cpu.yaml configs\my_work_cpu.yaml            # ואז לערוך 4 שורות בעותק
.\.venv\Scripts\python.exe tools\verify_clone.py --config configs\my_work_cpu.yaml
.\tools\run_work.ps1 -Step full -Config configs\my_work_cpu.yaml    # כל הריצה, בלי שאלות
# בסוף נדפס נתיב הדוח: פותחים את report.html בדפדפן
```

מריצים כל פקודה מתיקיית הפרויקט (התיקייה שמכילה את `src\`, `tools\`, `configs\`).

## מה זה עושה

המערכת לוקחת אודיו מתויג (קטעים של עד 30 שניות עם הטקסט שלהם), מפצלת לפי שיעור ל-Train / Val / Test, מאמנת את המפענח של המודל `ivrit-ai/whisper-large-v3-turbo` (המקודד קפוא והפלט שלו נשמר במטמון, כך שהאימון מהיר בערך פי 8 על המעבד), בודקת את המודל לפני האימון ואחריו, וכותבת דוח אחד שמראה אם האימון התכנס, אם היה ארוך מספיק, או אם הייתה התאמת יתר. קבוצת ה-Test אינה נפתחת אלא אם ביקשתם `-WithTest`. אין צורך באינטרנט או בהרשאות מנהל.

## מה צריך

- Python 3.12; Windows 11 (נבדק). לינוקס: `tools/run_work.sh` כתוב אך לא נבדק.
- זיכרון RAM: 16 ג'יגה מותקן; האימון מגיע לשיא של 6.4 עד 6.7 ג'יגה, לכן סוגרים תוכניות אחרות. דיסק: לפחות 30 ג'יגה פנויים. אין צורך ב-GPU.
- תיקיית המודל `models\whisper-large-v3-turbo-ivrit` (11 קבצים, 3.24 ג'יגה).
- תיקיית נתונים עם `manifest.csv` וקבצי האודיו (`docs\DATA_ONBOARDING.md`): עמודה לשם קובץ האודיו, עמודה לטקסט, ועמודת קבוצה (מזהה שיעור או הקלטה). הקטעים באורך של עד 30 שניות.

## התקנה

1. Python 3.12, והפרויקט בנתיב קצר (למשל `C:\stt`).
2. `python -m venv .venv` ואז `.\.venv\Scripts\Activate.ps1`.
3. עם אינטרנט: `pip install -r requirements.lock`. בלי אינטרנט: החבילות מגיעות מהתיקייה `wheels\` (היא לא ב-git, כ-230 מגה). מכינים אותה פעם אחת במחשב עם אינטרנט ואותה גרסת Python: `python tools\fetch_wheels.py --requirements requirements.txt --out wheels --lock requirements.lock --python-version 3.12 --platform win_amd64`, מעתיקים למחשב הסגור, ואז `pip install --no-index --find-links wheels -r requirements.lock`.
4. בדיקה: `.\.venv\Scripts\python.exe tools\verify_clone.py`.

## המודל והנתונים

- המודל: במחשב עם אינטרנט `python tools\fetch_model.py --repo ivrit-ai/whisper-large-v3-turbo --out models\whisper-large-v3-turbo-ivrit`, ומעתיקים את התיקייה לאותו מקום ב-clone.
- הנתונים: תיקייה כלשהי עם `manifest.csv` (UTF-8) והאודיו. עורכים ב-`configs\my_work_cpu.yaml` רק 4 שורות (תיקיית הנתונים ושלושת שמות העמודות; ראש הקובץ אומר אילו שורות).

## הפקודות והדוח

הפקודה `.\tools\run_work.ps1 -Step full -Config configs\my_work_cpu.yaml` מריצה הכול. אותה פקודה שוב ממשיכה אחרי עצירה או קריסה. `-Epochs 3` לשלושה סבבים. `-WithTest` מוסיף את בדיקת ה-Test החד-פעמית (רק בריצה הסופית). הדוח: `data\work_cpu\experiments\exp_NNNN\reports\report.html`. הסבר לכל מספר ולכל תקלה: `docs\WORK_RUNBOOK.md`.
