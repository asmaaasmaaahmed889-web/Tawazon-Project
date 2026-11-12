# 🌟 Tawazon (توازن) — AI-Powered Workplace Burnout Detection Platform
> **منصة ذكية متكاملة للتنبؤ بمخاطر الاحتراق الوظيفي بالاعتماد على خوارزميات التعلم الآلي والبيانات**  
> *مشروع تخرج تطبيقي متكامل (Graduation Project 2026)*  
> **فريق العمل:** عبدالله • أسماء • شيماء • أحمد • نورهان  
> **إشراف:** د. أمير العطار | د. ريهام مصطفى  
> **المعيدة:** م. أمل الشرقاوي

---

## 📌 الفهرس (Table of Contents)
1. [نظرة عامة على المشروع (Project Overview)](#1-نظرة-عامة-على-المشروع)
2. [معمارية الذكاء الاصطناعي (AI & Machine Learning Architecture)](#2-معمارية-الذكاء-الاصطناعي)
3. [المتغيرات التشغيلية المعتمدة (The 6 Workload Metrics)](#3-المتغيرات-التشغيلية-المعتمدة)
4. [توزيع المهام بين أعضاء الفريق (Team Roles & Responsibilities)](#4-توزيع-المهام-بين-أعضاء-الفريق)
5. [التقنيات المستخدمة (Tech Stack)](#5-التقنيات-المستخدمة)
6. [هيكل المشروع ومجلداته (Project Structure)](#6-هيكل-المشروع-ومجلداته)
7. [طريقة التثبيت والتشغيل (How to Run)](#7-طريقة-التثبيت-والتشغيل)
8. [حسابات الدخول التجريبية (Demo Credentials)](#8-حسابات-الدخول-التجريبية)
9. [واجهات البرمجة الخلفية (API Endpoints)](#9-واجهات-البرمجة-الخلفية)

---

## 1. نظرة عامة على المشروع
> **ملاحظة:** بناءً على توجيه المشرف، النموذج لا يستخدم تحليل النصوص (NLP). القرار النهائي يعتمد على المتغيرات الرقمية الستة فقط.

نظام **Tawazon (توازن)** هو حل مؤسسي ذكي مصمم لمساعدة إدارات الموارد البشرية والقيادات التنفيذية على التنبؤ بمخاطر **الاحتراق الوظيفي (Job Burnout)** في مراحله المبكرة. 

يعتمد النظام على **مقاييس تشغيلية (Workload Metrics)** لساعات العمل والإجازات وتراكم المهام كمدخلات لنموذج Random Forest، ويقدّم تقديراً لمستوى الخطر وتوصيات استباقية. النتيجة تقدير مبني على الارتباطات وليست تشخيصاً طبياً أو نفسياً، ولمزيد من التفاصيل راجع القسم 10 (حدود البيانات).

---

## 2. معمارية الذكاء الاصطناعي (AI Architecture)

* **الخوارزمية المعتمدة:** `Random Forest Classifier` (غابات عشوائية تجميعية مكونة من 200 شجرة قرار).
* **توازن الداتا (Data Balancing):** الداتا الأصلية كانت غير متوازنة (441 حالة إرهاق مقابل 559 بدون إرهاق). تم ضبطها إلى **400 حالة إرهاق + 400 حالة بدون إرهاق** (`data/employees_balanced.csv`) قبل إعادة التدريب.
* **النتائج النهائية (Random Forest، بيانات متوازنة، اختبار 160 موظف):**
  * **الدقة (Accuracy):** 90.0%
  * **Recall للإرهاق:** 93.8%
  * **F1-Score:** 90.4%
* **مقارنة قبل وبعد التوازن:** التفاصيل الكاملة في `models/model_comparison.csv`، ومصفوفتي الالتباس في `models/confusion_matrix_before.png` و`models/confusion_matrix_after.png`.
* **ملاحظة:** الدقة قبل التوازن (95.5%) كانت مضخّمة بسبب انحياز الداتا نحو فئة واحدة، لذلك النتيجة بعد التوازن هي الأدق للتقييم.
* **منهجية تقسيم البيانات (Data Splitting):**
  * `Train/Test Split (80% / 20%)`: 640 موظف للتدريب و 160 موظف للاختبار (بعد التوازن).
  * `Stratified Sampling`: للحفاظ على نسبة الاحتراق المتوازنة (44.1%) في كلا المجموعتين.
  * `random_state = 42`: لضمان تكرارية وثبات التجربة العلمية (Reproducibility).
* **الذكاء الاصطناعي القابل للتفسير (Explainable AI - XAI):**
  * استخدام قيم `SHAP (Shapley Additive Explanations)` لتوضيح العوامل الثلاثة الأولى التي رفعت مؤشر الخطر لكل موظف.

---

## 3. المتغيرات التشغيلية المعتمدة (The 6 Workload Metrics)

 تم استبعاد السن والجنس التزاماً بأخلاقيات الذكاء الاصطناعي (**AI Fairness**) لمنع التحيز التمييزي. 

المتغيرات الستة المعتمدة وترتيب أهميتها في القرار:
1. 🥇 **`overtime_hours_week` (30.7%):** ساعات العمل الإضافي أسبوعياً (المتسبب الأكبر في الإجهاد).
2. 🥈 **`vacation_days_last_year` (22.1%):** أيام الإجازة المستهلكة (مؤشر الاستشفاء والراحة).
3. 🥉 **`avg_daily_hours` (21.0%):** متوسط ساعات العمل اليومية الأساسية.
4. 🔹 **`late_arrivals_month` (16.2%):** مرات التأخر في الشهر.
5. 🔹 **`tasks_assigned` (6.5%):** عدد المهام المتزامنة النشطة.
6. 🔹 **`absence_days_month` (3.6%):** أيام الغياب الشهرية.

---

## 4. توزيع المهام بين أعضاء الفريق (Team Roles & Responsibilities)

تم تقسيم المشروع بصورة معيارية واحترافية ليغطي دورة حياة هندسة البرمجيات والذكاء الاصطناعي الكاملة بين 5 أعضاء:

### 👤 العضو الأول: مهندس تعلم الآلة وهندسة البيانات (Machine Learning & Data Engineer)
* **المسؤوليات:** معالجة مجموعة البيانات، تنظيف المدخلات، تقسيم البيانات (80/20 Stratified)، تدريب ومقارنة الخوارزميات (Logistic Regression vs Random Forest)، ضبط الـ Hyperparameters، وحفظ النموذج النهائي بصيغة binary.
* **الملفات المسندة:**
  * `train_model.py`
  * `data/employees.csv`
  * `data/generate_data.py`
  * `data/process_kaggle_data.py`
  * `models/workload_model.pkl`

---

### 👤 العضو الثاني: مهندس الذكاء الاصطناعي القابل للتفسير ومحرك المخاطر (XAI & Risk Engine Engineer)
* **المسؤوليات:** بناء محرك حساب مؤشر الخطر الرياضي (تحويل الاحتمالية إلى نسبة 0–100 وتحديد المستويات Low/Medium/High)، تطبيق خوارزمية SHAP لاستخراج مسببات الخطر الفردية، وتوليد مصفوفة الالتباس ورسوم الـ Feature Importance.
* **الملفات المسندة:**
  * `backend/risk.py`
  * `backend/explain.py`
  * `models/feature_importance.png`
  * `models/confusion_matrix.png`

---

### 👤 العضو الثالث: مهندس الخادم والواجهات البرمجية (Backend API & Security Engineer)
* **المسؤوليات:** بناء خادم FastAPI، تصميم مسارات RESTful API الكاملة (`/assess`, `/predict`, `/stats`, `/users`, `/auth`)، نماذج التحقق Pydantic Schemas، تأمين المصادقة وحماية كلمات المرور باستخدام JWT و Bcrypt، وإدارة جلسات SQLAlchemy ORM.
* **الملفات المسندة:**
  * `backend/main.py`
  * `backend/database.py`
  * `backend/auth.py`
  * `requirements.txt`

---

### 👤 العضو الرابع: مهندس الواجهة الأمامية وتجربة المستخدم (Frontend UI/UX Engineer)
* **المسؤوليات:** بناء وتصميم نظام المكونات البصري (Design System)، نظام التوطين والترجمة الفورية ثنائية اللغة (Arabic RTL / English LTR)، صفحات التقييم التفاعلية، النوافذ المنبثقة (Modals)، التنبيهات اللحظية (Toasts)، وتنسيقات الطباعة كتقارير PDF.
* **الملفات المسندة:**
  * `frontend/index.html`
  * `frontend/employees.html`
  * `frontend/assess.html`
  * `frontend/employee-portal.html`
  * `frontend/login.html`
  * `frontend/css/style.css`
  * `frontend/js/i18n.js`

---

### 👤 العضو الخامس: مهندس تحليلات البيانات والداشبورد وجودة النظام (Dashboard Analytics & QA Engineer)
* **المسؤوليات:** تكامل الرسوم البيانية التفاعلية الحية (Chart.js Donut & Benchmark Charts)، حساب مؤشرات الـ KPIs اللحظية، برمجة قائمة المتابعة الحرجة (Critical Watchlist)، مزامنة وتحديث قاعدة البيانات، وتجهيز العروض التقديمية والتوثيق الأكاديمي.
* **الملفات المسندة:**
  * `frontend/js/app.js`
  * `frontend/js/auth.js`
  * `frontend/js/config.js`
  * `Tawazon_Defense_Presentation.pptx`
  * `tools/sync_database_scores.py`

---

## 5. التقنيات المستخدمة (Tech Stack)

* **Artificial Intelligence & ML:** `Python 3.10+`, `Scikit-Learn`, `SHAP`, `Pandas`, `NumPy`, `Joblib`.
* **Backend & API:** `FastAPI`, `Uvicorn`, `Pydantic v2`, `SQLAlchemy`, `Bcrypt`, `Python-Multipart`.
* **Database:** `SQLite` (خفيفة وسريعة وجاهزة للترقية إلى PostgreSQL).
* **Frontend:** `HTML5`, `CSS3 (Modern Custom Design System)`, `Vanilla Modern JavaScript (ES6+)`, `Chart.js 4`.
* **Security & Auth:** `JWT Bearer Tokens`, `Password Hashing with Salt`.

---

## 6. هيكل المشروع ومجلداته (Project Structure)

```text
Tawazon/
├── backend/                  # خادم الباك إند ومنطق الذكاء الاصطناعي
│   ├── auth.py               # نظام المصادقة والصلاحيات (JWT + Bcrypt)
│   ├── database.py           # نماذج قاعدة البيانات (SQLAlchemy ORM)
│   ├── explain.py            # محرك تفسير القرارات (SHAP Explainability)
│   ├── main.py               # نقاط اتصال الواجهة البرمجية (FastAPI App)
│   └── risk.py               # محرك حساب مؤشرات ومستويات الخطر
├── data/                     # مجموعات البيانات وسكريبتات التجهيز
│   ├── employees.csv         # مجموعة بيانات 1,000 موظف بالمتغيرات الـ 6
│   ├── generate_data.py      # سكريبت توليد وتخليق البيانات
│   └── process_kaggle_data.py# سكريبت معالجة وتجهيز بيانات Kaggle
├── frontend/                 # الواجهة الأمامية التفاعلية
│   ├── assets/               # شعارات وصور النظام
│   ├── css/style.css         # نظام التصميم الموحد والتنسيقات
│   ├── js/
│   │   ├── app.js            # منطق لوحة التحكم والرسوم البيانية
│   │   ├── auth.js           # إدارة جلسات تسجيل الدخول بالمتصفح
│   │   ├── config.js         # إعدادات رابط السيرفر
│   │   └── i18n.js           # محرك الترجمة الفورية عربي/إنجليزي
│   ├── assess.html           # شاشة إجراء تقييم إرهاق جديد
│   ├── employee-portal.html  # بوابة الموظف الذاتية
│   ├── employees.html        # دليل وإدارة شؤون الموظفين
│   ├── index.html            # لوحة التحليلات التنفيذية المتقدمة
│   └── login.html            # شاشة تسجيل الدخول
├── models/                   # النماذج المدربة ورسوم التقييم
│   ├── workload_model.pkl    # نموذج Random Forest المحفوظ
│   ├── feature_importance.png
│   └── confusion_matrix.png
├── tools/                    # أدوات مساعدة للصيانة
│   └── sync_database_scores.py  # تحديث درجات الخطر المخزنة بعد إعادة تدريب النموذج
├── train_model.py            # سكريبت تدريب وتقسيم وتقييم النموذج
├── requirements.txt          # حزم ومكتبات بايثون المطلوبة
├── Tawazon_Defense_Presentation.pptx # العرض التقديمي للمناقشة
└── README.md                 # دليل المشروع الكامل
```

---

## 7. طريقة التشغيل السريعة على ويندوز (Windows)

1. ثبّتي **Python 3.13** من موقع python.org (مش من Microsoft Store)، واختاري **Add python.exe to PATH** أثناء التثبيت.
2. افتحي فولدر المشروع، ودبل كليك على **setup.bat** (مرة واحدة بس، وبيأخد بضع دقايق).
3. كل مرة تشغلي فيها المشروع، دبل كليك على **run.bat**.
4. افتحي `frontend/login.html` في المتصفح.

ملاحظة: أول تشغيل بيجهز حسابات الموظفين وبياخد حوالي 5 دقايق.

## 7. طريقة التثبيت والتشغيل (How to Run)

### المتطلبات الأساسية:
تثبيت بيئة `Python 3.10` أو أحدث.

### 1. تثبيت الحزم المطلوبة:
```bash
pip install -r requirements.txt
```

### 2. تدريب النموذج (اختياري، النموذج محفوظ وجاهز):
```bash
python train_model.py
```

### 3. تشغيل خادم الباك إند (FastAPI):
```bash
uvicorn backend.main:app --reload
```
يعمل السيرفر افتراضياً على: `http://127.0.0.1:8000`  

### 4. تشغيل الموقع (Frontend):
افتح أي صفحة من مجلد `frontend/` (مثل `frontend/index.html` أو `frontend/login.html`) مباشرة في المتصفح، أو عبر Live Server.

---

## 8. حسابات الدخول التجريبية (Demo Credentials)

| الدور (Role) | اسم المستخدم (Username) | كلمة المرور (Password) | الصلاحيات |
| :--- | :--- | :--- | :--- |
| **مسؤول النظام (Admin)** | `admin` | `*********` | الوصول للداشبورد التنفيذية، إدارة الموظفين، والتقارير |
---

## 9. واجهات البرمجة الخلفية (API Endpoints)

* `POST /auth/login`: تسجيل الدخول واستخراج توكن الصلاحية.
* `GET /stats`: إحصائيات الداشبورد، نسب توزيع المخاطر، وقائمة المتابعة العاجلة.
* `GET /employees`: استرجاع قائمة الموظفين مع البحث والتصفية وتقسيم الصفحات (Pagination).
* `POST /assess`: تسجيل تقييم إرهاق جديد وحساب الخطر وتوليد الأسباب فورياً عبر ML.
* `POST /predict`: تنبؤ فوري سريع دون حفظ في قاعدة البيانات.
* `GET /employee/profile`: استرجاع بيانات وسجل تقييمات الموظف المسجل حالياً.

---

## 10. Limitations and Data Disclosure

The workload model was trained and evaluated on the `employees.csv` dataset, which was derived from the Kaggle dataset "Are Your Employees Burning Out?" (Varshney, 2020). The six workload features (average daily hours, overtime, absence days, late arrivals, tasks assigned, and vacation days) were generated from the original Burn Rate score, and the burnout label was defined as Burn Rate ≥ 0.5. Consequently, the model learns the relationship used to construct the dataset rather than an independently measured relationship between workload and burnout. The reported performance (Accuracy 90.0%, Recall 93.8%, F1 90.4% after balancing) therefore measures how well the model recovers this construction rule, and should not be interpreted as evidence of its ability to predict burnout in real workplaces.

The burnout label is not based on a validated burnout instrument. Additionally, the test set contains 160 employees, which limits the precision of the estimates. The model has not been validated on external or real-world employee data. Real-world validation, using a recognized burnout measure such as the Maslach Burnout Inventory or the Copenhagen Burnout Inventory, is required before the system is used for decision-making.
