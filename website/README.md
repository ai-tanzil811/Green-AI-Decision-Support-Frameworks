# GreenPEFT Website Directory

This directory contains the self-contained static onboarding landing website for **GreenPEFT**.

## Structure

```text
website/
├── index.html        # Main landing page HTML5 structure
├── styles.css        # Modern Dark Eco-Tech styling system
├── script.js          # Interactive GEI simulator & copy clipboard logic
├── methodology.png   # Methodology pipeline diagram
└── netlify.toml      # Netlify site configuration
```

## Deployment Instructions

### 1. Drag & Drop on Netlify
Drag and drop this `website/` folder directly into [Netlify Drop](https://app.netlify.com/drop).

### 2. GitHub Continuous Deployment
If deploying the parent repository on Netlify, the root `netlify.toml` is pre-configured to publish this `website` directory automatically on every push!
