# ResumeIQ

### AI-Powered Resume Screening & Interview Assistant

ResumeIQ is an AI-powered resume analysis application that evaluates how well a candidate's resume matches a specific job description and provides personalized improvement feedback and interview preparation.

The application combines **Azure AI Document Intelligence**, **Microsoft Foundry GPT-4.1-mini**, **Azure Blob Storage**, and **Azure Functions** to create an end-to-end AI-assisted recruitment and career preparation workflow.

---

## 🚀 Features

### 📊 ATS Resume Analysis

Upload a resume and provide a job description to receive an AI-generated compatibility analysis.

The system provides:

- Overall Match Score
- Keyword Match score
- Experience Alignment score
- Skills Coverage score
- Matched skills and keywords
- Missing skills and keywords

The ATS evaluation uses a structured scoring system:

| Category | Weight |
|---|---:|
| Keyword Match | 40 |
| Experience Alignment | 30 |
| Skills Coverage | 30 |
| **Total** | **100** |

---

### ✨ AI Feedback

ResumeIQ goes beyond simply providing a score.

The AI analyzes the ATS evaluation and provides:

- A concise resume assessment
- Specific areas for improvement
- Practical recommendations
- A single recommended action

The feedback is displayed inside a dedicated modal so the main analysis interface remains clean.

---

### 🧠 AI Interview Prep

ResumeIQ can generate interview questions specifically tailored to:

- The candidate's resume
- The target job description
- The technologies and skills relevant to the role

The system generates **5 personalized interview questions** covering areas such as:

- Technical skills
- Projects
- Technologies
- Resume experience
- Role-specific knowledge

---

### 💡 Suggested Interview Answers

Each generated interview question includes an optional suggested answer.

Answers are designed to:

- Be based on information present in the candidate's resume
- Match the target role
- Provide a useful starting point for interview preparation
- Avoid inventing experience or qualifications

Answers can be revealed individually using the **Show Suggested Answer** button.

---

## 🏗️ Architecture

```text
                    Resume + Job Description
                              │
                              ▼
                    ┌───────────────────┐
                    │  Azure Blob       │
                    │  Storage          │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Azure AI          │
                    │ Document          │
                    │ Intelligence      │
                    └─────────┬─────────┘
                              │
                         Resume Text
                              │
                              ▼
                    ┌───────────────────┐
                    │ Microsoft Foundry │
                    │ GPT-4.1-mini      │
                    └─────────┬─────────┘
                              │
              ┌───────────────┼────────────────┐
              │               │                │
              ▼               ▼                ▼
        ATS Analysis      AI Feedback     Interview Prep
              │               │                │
              ▼               ▼                ▼
        Match Report      Improvements    Questions +
                                          Suggested
                                           Answers 