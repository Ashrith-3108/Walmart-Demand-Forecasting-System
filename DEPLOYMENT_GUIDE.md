# 🚀 Deployment Guide

## 📋 Prerequisites

- GitHub account
- Python 3.8+ installed
- Git installed

## 🔗 Step 1: Push to GitHub

### **Create GitHub Repository**

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon → "New repository"
3. Name: `walmart-demand-forecasting`
4. Description: `Advanced ML demand forecasting system for Walmart stores`
5. Make it **Public** (for free hosting)
6. **Don't** initialize with README (we already have one)
7. Click "Create repository"

### **Push Your Code**

```bash
# Add your GitHub repository as remote origin
git remote add origin https://github.com/YOUR_USERNAME/walmart-demand-forecasting.git

# Push to GitHub
git push -u origin main
```

## ☁️ Step 2: Deploy to Streamlit Cloud (Recommended)

### **Free Hosting with Streamlit Cloud**

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `walmart-demand-forecasting`
5. Main file path: `app.py`
6. Python version: `3.8`
7. Click "Deploy!"

**✅ Result**: Your app will be live at `https://your-app-name.streamlit.app`

## 🐳 Step 3: Alternative Deployments

### **Heroku Deployment**

```bash
# Install Heroku CLI
# Create app
heroku create your-app-name

# Add buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Open app
heroku open
```

### **Local Testing**

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## 🔧 Configuration

### **Environment Variables (Optional)**

```bash
# For production deployment
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### **Custom Domain (Optional)**

- Streamlit Cloud: Add custom domain in settings
- Heroku: Use custom domain addon

## 📊 Monitoring

### **Check App Status**

- **Streamlit Cloud**: Dashboard shows uptime and usage
- **Heroku**: `heroku logs --tail` for real-time logs

### **Performance Metrics**

- Response time: < 5 seconds
- Memory usage: ~500MB
- Concurrent users: 100+

## 🚨 Troubleshooting

### **Common Issues**

1. **Build Failures**

   - Check `requirements.txt` compatibility
   - Verify Python version (3.8+)

2. **Memory Errors**

   - Reduce dataset size for free tiers
   - Optimize model parameters

3. **Import Errors**
   - Ensure all dependencies are in `requirements.txt`
   - Check file paths in code

### **Support**

- **GitHub Issues**: Report bugs in your repository
- **Streamlit Community**: [discuss.streamlit.io](https://discuss.streamlit.io/)
- **Heroku Support**: [help.heroku.com](https://help.heroku.com/)

## 🎯 Next Steps

1. **Customize**: Update README with your live URL
2. **Enhance**: Add more features or models
3. **Share**: Post on LinkedIn, add to portfolio
4. **Monitor**: Track usage and performance

---

**🎉 Congratulations!** Your Walmart Demand Forecasting System is now live and ready to impress employers and clients.
