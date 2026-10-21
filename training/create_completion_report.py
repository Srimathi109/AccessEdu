from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results' / 'AccessEdu_Project_Completion_Report_Updated.docx'
GREEN, DARK, PALE, LIGHT, MUTED, WHITE = '176B45', '173328', 'EAF3EC', 'F5F8F5', '607168', 'FFFFFF'

def shade(cell, color):
    pr = cell._tc.get_or_add_tcPr()
    node = pr.find(qn('w:shd'))
    if node is None:
        node = OxmlElement('w:shd'); pr.append(node)
    node.set(qn('w:fill'), color)

def cell_text(cell, text, bold=False, color=DARK, size=8.8):
    cell.text = ''
    p = cell.paragraphs[0]; p.paragraph_format.space_after = Pt(0)
    r = p.add_run(str(text)); r.bold = bold; r.font.name = 'Arial'; r.font.size = Pt(size); r.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

def table(doc, headers, rows, widths):
    t = doc.add_table(rows=1, cols=len(headers)); t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
    for i, h in enumerate(headers): t.rows[0].cells[i].text = h
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row): cells[i].text = str(value)
    for ri, row in enumerate(t.rows):
        trpr = row._tr.get_or_add_trPr(); trpr.append(OxmlElement('w:cantSplit'))
        if ri == 0:
            repeat = OxmlElement('w:tblHeader'); repeat.set(qn('w:val'), 'true'); trpr.append(repeat)
        for ci, cell in enumerate(row.cells):
            cell.width = Inches(widths[ci])
            if ri == 0: shade(cell, GREEN); cell_text(cell, cell.text, True, WHITE, 8.4)
            else:
                if ri % 2 == 0: shade(cell, LIGHT)
                cell_text(cell, cell.text)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return t

def para(doc, text, after=6):
    p = doc.add_paragraph(text); p.paragraph_format.space_after = Pt(after); p.paragraph_format.line_spacing = 1.08
    return p

def heading(doc, text, level=1):
    p = doc.add_heading(text, level); p.paragraph_format.keep_with_next = True
    return p

def bullet(doc, text, numbered=False):
    p = doc.add_paragraph(style='List Number' if numbered else 'List Bullet')
    p.paragraph_format.space_after = Pt(3); p.add_run(text); return p

def callout(doc, title, text):
    t = doc.add_table(rows=1, cols=1); c = t.cell(0, 0); shade(c, PALE); c.text = ''
    p = c.paragraphs[0]; p.paragraph_format.space_after = Pt(4)
    r = p.add_run(title); r.bold = True; r.font.name = 'Arial'; r.font.size = Pt(10); r.font.color.rgb = RGBColor.from_string(GREEN)
    p = c.add_paragraph(text); p.paragraph_format.space_after = Pt(0)
    for r in p.runs: r.font.name = 'Arial'; r.font.size = Pt(9); r.font.color.rgb = RGBColor.from_string(DARK)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)

doc = Document()
sec = doc.sections[0]; sec.top_margin=Inches(.72); sec.bottom_margin=Inches(.68); sec.left_margin=Inches(.82); sec.right_margin=Inches(.82)
normal=doc.styles['Normal']; normal.font.name='Arial'; normal.font.size=Pt(9.5); normal.font.color.rgb=RGBColor.from_string(DARK)
for name, size, color in [('Title', 28, DARK), ('Heading 1', 17, DARK), ('Heading 2', 12, GREEN)]:
    s=doc.styles[name]; s.font.name='Arial'; s.font.size=Pt(size); s.font.bold=True; s.font.color.rgb=RGBColor.from_string(color); s.paragraph_format.space_before=Pt(10); s.paragraph_format.space_after=Pt(6)
for name in ['List Bullet','List Number']:
    doc.styles[name].font.name='Arial'; doc.styles[name].font.size=Pt(9.3)

# Cover
p=doc.add_paragraph(); p.paragraph_format.space_before=Pt(76); p.paragraph_format.space_after=Pt(12)
r=p.add_run('RESEARCH PROJECT'); r.bold=True; r.font.name='Arial'; r.font.size=Pt(10); r.font.color.rgb=RGBColor.from_string(GREEN)
p=doc.add_paragraph(style='Title'); p.add_run('AccessEdu Project Completion Report')
p=doc.add_paragraph(); p.paragraph_format.space_before=Pt(8); p.paragraph_format.space_after=Pt(25)
r=p.add_run('Cricket Umpire Signal Practice and Assessment System'); r.font.name='Arial'; r.font.size=Pt(15); r.font.color.rgb=RGBColor.from_string(MUTED)
t=table(doc,['REPORT DETAILS',''],[
 ('Project status','Functional local demonstration completed'),
 ('Project area','Computer vision and sports umpire training'),
 ('Report date','20 October 2026'),
 ('Evaluation','Original checkpoint compared with an augmented candidate'),
],[1.8,4.8])
for row in t.rows[1:]:
    shade(row.cells[0],PALE)
    for r in row.cells[0].paragraphs[0].runs: r.bold=True; r.font.color.rgb=RGBColor.from_string(GREEN)
p=doc.add_paragraph(); p.paragraph_format.space_before=Pt(24); p.paragraph_format.space_after=Pt(4)
r=p.add_run('COMPLETION SUMMARY'); r.bold=True; r.font.name='Arial'; r.font.size=Pt(9); r.font.color.rgb=RGBColor.from_string(GREEN)
para(doc,'AccessEdu provides a local practice workflow in which trainees record umpire signals, receive model-generated feedback, and review saved attempts. Instructors can create scenarios, add trainee accounts, and inspect progress summaries. Evaluation found useful signal detection, but the evidence does not justify guaranteeing correct predictions. The app remains a demonstration and practice aid, not a certification or match-officiating system.')
doc.add_page_break()

heading(doc,'Executive Summary')
para(doc,'The project addresses a practical training need: trainee cricket umpires need a repeatable way to practise signals and review their consistency. AccessEdu connects the cricket gesture detector to a small local web application. A trainee chooses a scenario, records or uploads a short video, and receives a comparison between the expected signal and the most frequently detected signal. Instructors can create scenarios and review saved trainee summaries.')
para(doc,'The project is complete as a functional prototype. It includes role-based login, scenario management, browser video recording and upload, sampled-frame YOLO inference, saved practice attempts, and an instructor overview. A synthetic stress set was generated from the training split, and a fine-tuned candidate was compared with the original checkpoint. The candidate improved on synthetic variants but scored lower on the original validation and test splits, so it was not promoted.')
callout(doc,'Main conclusion','The demo shows how computer vision can support structured signal practice. It does not establish dependable real-world accuracy: the independent test split contains only nine images and no Out examples, while the larger synthetic set is derived from training images.')

heading(doc,'Project Aim and Objectives')
para(doc,'The project aim is to provide trainee cricket umpires with a simple tool for practising standardized signals and receiving immediate, repeatable feedback. The working research question is: How can computer vision support self-guided practice of cricket umpire signals, and what evidence is needed before that feedback can be trusted?')
for x in [
 'Build a trainee workflow for selecting a signal scenario and submitting a practice recording.',
 'Use the cricket object-detection model to recognize signal categories in sampled video frames.',
 'Compare each prediction with the scenario expectation and save a practice summary.',
 'Give instructors tools to create scenarios, add trainee accounts, and review progress summaries.',
 'Compare the original checkpoint and an augmented-data candidate while distinguishing synthetic and independent evaluation.',
]: bullet(doc,x)

heading(doc,'Development Approach: Five Sprints')
para(doc,"The project was organized into five sprints. The sprint themes below reflect the team's Scrum plan; sprint dates, individual task assignments, and per-sprint completion metrics were not provided.")
table(doc,['Sprint','Focus','Work area'],[
 ('Sprint 1','Research & Requirements','Define the training problem, intended users, scope, and requirements.'),
 ('Sprint 2','Dataset & ML Model','Prepare the cricket signal dataset and develop and evaluate the detection model.'),
 ('Sprint 3','Real-Time Recognition','Connect model inference to video-frame processing for signal recognition.'),
 ('Sprint 4','Accessibility & Event Interpretation','Shape the user experience around accessible practice and interpreting detected signal events.'),
 ('Sprint 5','AccessEdu Application','Integrate the application workflow into the AccessEdu application.'),
],[0.8,2.35,3.25])

heading(doc,'System Workflow')
for x in [
 'An instructor signs in, creates or activates scenarios, and creates trainee accounts.',
 'A trainee signs in and chooses an active signal scenario.',
 'The trainee records a short browser clip or uploads a supported video file.',
 'The application samples up to 24 frames and runs the cricket YOLO model on them.',
 'AccessEdu selects the most frequently detected signal, compares it with the expected signal, and displays feedback.',
 'The attempt is saved for trainee history and instructor progress summaries.',
]: bullet(doc,x,True)

heading(doc,'Implemented Components')
table(doc,['Component','Implementation','Purpose'],[
 ('Web application','Flask and server-rendered HTML templates','Local role-based trainee and instructor workflows'),
 ('Video capture','Browser MediaRecorder and file upload','Capture a practice clip without a separate React application'),
 ('Inference','Ultralytics YOLO and OpenCV','Classify No_Ball, Out, Six, and Wide'),
 ('Persistence','SQLite','Store users, scenarios, and attempt summaries'),
 ('Training assets','Roboflow YOLO dataset and checkpoints','Train and evaluate the cricket signal detector'),
],[1.25,2.65,2.7])

heading(doc,'Data and Model')
para(doc,'The current application supports four cricket umpire signals: No_Ball, Out, Six, and Wide. It samples video frames with an image classifier and aggregates class probabilities. Its feedback concerns signal category and confidence, not match-decision correctness or signal technique.')
para(doc,'Five deterministic variants were created for each training image: horizontal flip, exposure shift, small rotation/scale/translation with adjusted bounding boxes, reduced-resolution blur, and sensor noise. This produced 825 synthetic images with labels. Because each variant comes from a training image, this is a stress check and cannot serve as independent test evidence.')

heading(doc,'Evaluation Method')
para(doc,'The original checkpoint was evaluated on the original validation and test splits and on the 825-image synthetic stress set. A candidate checkpoint was fine-tuned for five epochs using original training images plus the synthetic variants. Both checkpoints were then evaluated at 640-pixel inference size on the same three evaluation sets. Results are object-detection mean average precision at intersection-over-union 0.50 (mAP50), with mAP50-95 shown as a stricter localization measure. These are not whole-video signal classification accuracy scores.')

heading(doc,'Model Evaluation Results')
table(doc,['Checkpoint','Validation\n15 images','Held-out test\n9 images','Synthetic stress\n825 images'],[
 ('Original mAP50','0.879','0.995','0.724'),
 ('Augmented candidate mAP50','0.845','0.953','0.803'),
 ('Original mAP50-95','0.731','0.870','0.562'),
 ('Augmented candidate mAP50-95','0.608','0.719','0.604'),
],[2.0,1.45,1.45,1.7])
para(doc,'The candidate raised synthetic mAP50 by about 0.079, but lowered mAP50 on the original validation and held-out test splits. It also lowered mAP50-95 on both original splits. The original test result is based on only nine labeled objects and does not cover Out, so its near-perfect mAP50 should not be interpreted as reliable general performance. The candidate was not selected because it regressed on the original held-out evidence.')

heading(doc,'Completion Assessment')
table(doc,['Area','Status','Evidence'],[
 ('Trainee practice flow','Implemented','Scenario, video, prediction, result summary'),
 ('Instructor tools','Implemented','Scenario management, trainee creation, progress overview'),
 ('Saved progress','Implemented','SQLite attempt history and instructor summaries'),
 ('Model connection','Implemented','Cricket checkpoint used in sampled-frame video inference'),
 ('Independent accuracy evidence','Incomplete','Only 9 test images; no Out examples in this split'),
 ('Production deployment','Out of scope','Local prototype; no production authentication or deployment'),
],[1.6,1.5,3.5])

heading(doc,'Limitations')
for x in [
 'The dataset is too small to estimate reliability across different umpires, cameras, lighting, distances, and backgrounds.',
 'Synthetic stress images are derived from training data and cannot replace independently recorded test videos.',
 'The detector does not assess hand position, timing, signal clarity, decision correctness, or certification readiness.',
 'Automated feedback can be wrong and should not be treated as an official assessment without instructor validation.',
 'The app is configured for local demonstration; demo credentials and local SQLite storage are not a production identity or deployment system.',
]: bullet(doc,x)

heading(doc,'Conclusion and Next Steps')
para(doc,'AccessEdu is complete as a local research demonstration of a computer-vision-supported umpire practice workflow. It connects the existing cricket detector to trainee practice, stores outcomes, and gives instructors a progress view. The evaluation also establishes a clear boundary: the model can produce plausible signal detections, but this project does not yet have enough independent evidence to promise correct predictions.')
for x in [
 'Collect and label new videos from multiple umpires for every class, especially Out, and keep them separate from training data.',
 'Evaluate whole-video signal decisions and per-class errors on a larger instructor-reviewed test set.',
 'Review false positives, false negatives, and confidence thresholds with qualified umpire instructors.',
 'Promote a new checkpoint only when it improves on independent recordings without unacceptable class-specific regressions.',
 'Before deployment beyond a local demo, replace demo accounts and add production authentication, security controls, and data-retention policies.',
]: bullet(doc,x,True)

heading(doc,'Reproducibility and Project Files')
para(doc,'Run the local application from the project root using the setup instructions in README.md. Regenerate the synthetic stress set with python training/generate_synthetic_stress.py. The augmented fine-tuning configuration is dataset/augmented_training.yaml. Evaluation details are in results/model_robustness_report.md, with model run artifacts under runs/detect. Dataset attribution is documented in README.dataset.txt and README.roboflow.txt.')
heading(doc,'Reference')
para(doc,'Cricket Umpire Action Detection dataset, Roboflow Universe, CC BY 4.0. Dataset source: https://universe.roboflow.com/prabu/cricket-umpire-action-detection/dataset/1. Export details and attribution are recorded in README.roboflow.txt.')

doc.core_properties.title='AccessEdu Project Completion Report'
doc.core_properties.subject='Cricket Umpire Signal Practice and Assessment System'
doc.core_properties.author='AccessEdu Project Team'
doc.core_properties.keywords='cricket, umpire, computer vision, YOLO, project completion'
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
doc.save(OUTPUT)
print(OUTPUT)
