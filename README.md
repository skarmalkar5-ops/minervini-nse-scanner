# Minervini NSE Scanner

One-click Streamlit scanner:

NSE → Trend Template → RS → Leader Score → VCP proxy → CSV → Email

## Deploy

1. Create a GitHub repository.
2. Add `app.py` and `requirements.txt`.
3. Deploy the repository on Streamlit Community Cloud.
4. Add SMTP credentials as Streamlit Secrets.

## Required Streamlit secrets

```toml
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "your-email@gmail.com"
SMTP_PASSWORD = "your-app-password"
EMAIL_TO = "your-email@gmail.com"
```

Never commit passwords or app passwords to GitHub.

## Current screening rules

- Trend Score >= 88.9% (8/9)
- RS Rating >= 80
- Leader Score >= 85
- VCP Score >= 80
- No arbitrary top-50 limit

The system is a research/screening tool, not an automatic trading system.
