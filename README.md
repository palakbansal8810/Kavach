##  **Setup and Usage of main.py**

### 1. **Clone the Repository**
```bash
git clone [https://github.com/your-repo/kavach.git](https://github.com/palakbansal8810/Kavach.git)
cd kavach
```

### 2. **Install Required Packages**
```bash
pip install -r requirements.txt
```

### 3. **Set Up Environment Variables**
Create a `.env` file in the project root and add:
```
HF_TOKEN=your_huggingface_api_token
GROQ_API_KEY=your_groq_api_key
```

### 4. **Run the Application**
```bash
uvicorn main:app --reload
```
- Visit: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 📚 **Usage Instructions**

1. **Open the App:**  
   - Access `http://127.0.0.1:8000` to interact with Kavach.

2. **Ask a Question:**  
   - Type your query and get safety-related answers.

---

## 🔥 **API Endpoint**

### **Ask a Question**
```
POST /ask/
```
- **Request:**
```json
{
  "input": "Your question here"
}
```
- **Response:**
```json
{
  "answer": "AI-generated response"
}
```



##  **Setup and Usage of app.py**




### 1. **Run the FastAPI Application**
```bash
uvicorn app:app --reload
```

### 2. **Access the Application**
```bash
http://127.0.0.1:8000
```

---

## 📚 **API Usage**

### 1. **Test with curl**
```bash
curl -X POST "http://127.0.0.1:8000/ask/" -H "Content-Type: application/json" -d '{"input": "How can I stay safe while traveling at night?"}'
```

### 2. **Response**
```json
{
  "answer": "Avoid poorly lit areas, stay in populated places, and keep emergency contacts handy."
}
```
