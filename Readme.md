# Clone
git clone https://github.com/YOUR_USERNAME/texai.git
cd texai

# Create venv and activate
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
cd backend
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install inference-sdk

# Run backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000