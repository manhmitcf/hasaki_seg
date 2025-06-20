# 🔍 Products Finder - Tìm kiếm sản phẩm thông minh với AI

Một ứng dụng tìm kiếm sản phẩm sử dụng công nghệ AI và Vector Search với khả năng Text Chunking để tăng độ chính xác tìm kiếm.

## ✨ Tính năng chính

- 🤖 **Tìm kiếm ngữ nghĩa**: Sử dụng mô hình AI Vietnamese-BiEncoder để hiểu ý nghĩa câu truy vấn
- 🧩 **Text Chunking**: Chia nhỏ mô tả dài thành các đoạn ngắn để tìm kiếm chính xác hơn
- ⚡ **Vector Search**: Tìm kiếm nhanh chóng với MongoDB Atlas Vector Search
- 🌐 **API RESTful**: FastAPI với documentation tự động
- 📊 **Giao diện web**: Streamlit app với UI thân thiện
- 📈 **Phân tích kết quả**: Hiển thị điểm số, chunks và thống kê

## 🚀 Cài đặt và chạy

### 1. Clone repository và cài đặt dependencies

```bash
git clone <repository-url>
cd products-finder-python
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Cấu hình MongoDB Atlas

Tạo file `.env` trong thư mục gốc:

```env
MONGO_USER=your_username
MONGO_PASS=your_password
MONGO_HOST=your_cluster.mongodb.net
MONGO_DB=your_database_name
MONGO_COLLECTION=your_collection_name
```

### 3. Chuẩn bị dữ liệu

```bash
# Chạy script setup dữ liệu (Windows)
setup_data.bat

# Hoặc chạy thủ công
python load_data.py
```

### 4. Chạy ứng dụng

#### Cách 1: Sử dụng script (Windows)
```bash
run_app.bat
```

#### Cách 2: Chạy thủ công

**Terminal 1 - API Server:**
```bash
python main.py
```

**Terminal 2 - Streamlit UI:**
```bash
streamlit run streamlit_app.py
```

### 5. Truy cập ứng dụng

- **Web UI**: http://localhost:8501
- **API Documentation**: http://localhost:8001/docs
- **API Base URL**: http://localhost:8001

## 📁 Cấu trúc project

```
products-finder-python/
├── data/
│   └── products_data.json          # Dữ liệu sản phẩm
├── main.py                         # FastAPI server chính
├── streamlit_app.py               # Giao diện web Streamlit
├── text_chunker.py                # Module xử lý text chunking
├── load_data.py                   # Script tải dữ liệu lên MongoDB
├── requirements.txt               # Dependencies Python
├── setup_data.bat                 # Script setup dữ liệu (Windows)
├── run_app.bat                    # Script chạy app (Windows)
├── .env                          # Cấu hình môi trường (tạo thủ công)
└── README.md                     # File này
```

## 🔧 API Endpoints

### POST `/search`
Tìm kiếm sản phẩm với text chunking

**Request Body:**
```json
{
  "text": "sữa rửa mặt cho da dầu",
  "limit": 5,
  "chunk_limit": 20
}
```

**Response:**
```json
[
  {
    "product_id": "123",
    "name": "Sữa rửa mặt Cetaphil",
    "brand": "Cetaphil",
    "price": 250000,
    "score": 0.85,
    "descriptioninfo": "Sữa rửa mặt dành cho da dầu...",
    "total_chunks_found": 3,
    "relevant_chunks": ["chunk1", "chunk2", "chunk3"]
  }
]
```

### GET `/info`
Lấy thông tin về dữ liệu chunked

### POST `/search-chunks`
Tìm kiếm trực tiếp chunks (dành cho debug)

## 🧩 Text Chunking

### Cách hoạt động
1. **Chia nhỏ văn bản**: Mô tả dài được chia thành các đoạn 300 ký tự
2. **Overlap**: Các đoạn có phần chồng lấp 50 ký tự để đảm bảo ngữ cảnh
3. **Vector hóa**: Mỗi chunk được chuyển thành vector riêng biệt
4. **Tìm kiếm**: Tìm kiếm trên tất cả chunks
5. **Gộp kết quả**: Gộp các chunks cùng sản phẩm thành kết quả cuối

### Lợi ích
- ✅ Tìm kiếm chính xác hơn trong mô tả dài
- ✅ Tăng tốc độ xử lý
- ✅ Giảm nhiễu từ thông tin không liên quan
- ✅ Hiển thị đoạn văn bản liên quan nhất

## 🎯 Cách sử dụng

### 1. Giao diện web (Streamlit)
1. Mở http://localhost:8501
2. Nhập từ khóa tìm kiếm
3. Xem kết quả với thông tin chunks
4. Click vào link để xem chi tiết sản phẩm

### 2. API (FastAPI)
```python
import requests

response = requests.post(
    "http://localhost:8001/search",
    json={
        "text": "kem chống nắng SPF 50",
        "limit": 5,
        "chunk_limit": 20
    }
)

results = response.json()
```

### 3. Ví dụ tìm kiếm
- "sữa rửa mặt cho da dầu"
- "kem chống nắng SPF 50"
- "serum vitamin C chống lão hóa"
- "mặt nạ dưỡng ẩm ban đêm"
- "toner cho da nhạy cảm"

## 📊 Monitoring và Debug

### Kiểm tra trạng thái
- **API Health**: GET http://localhost:8001/docs
- **Chunking Info**: GET http://localhost:8001/info
- **Debug Chunks**: POST http://localhost:8001/search-chunks

### Logs và Metrics
- Thời gian tìm kiếm
- Số chunks tìm thấy
- Điểm tương đồng
- Thống kê chunks per product

## 🛠️ Troubleshooting

### Lỗi thường gặp

**1. Không kết nối được MongoDB**
```
Solution: Kiểm tra file .env và thông tin kết nối
```

**2. Model không tải được**
```
Solution: Đảm bảo có internet để tải model lần đầu
```

**3. API không phản hồi**
```
Solution: Kiểm tra port 8001 có bị chiếm không
```

**4. Streamlit không hiển thị**
```
Solution: Kiểm tra port 8501 và chạy lại streamlit
```

### Performance Tuning

**Tăng tốc độ:**
- Giảm `chunk_limit` trong request
- Tăng `numCandidates` trong vector search
- Sử dụng SSD cho MongoDB

**Tăng độ chính xác:**
- Giảm `chunk_size` trong TextChunker
- Tăng `overlap` giữa các chunks
- Fine-tune model cho domain cụ thể

## 🔄 Cập nhật dữ liệu

### Thêm sản phẩm mới
1. Cập nhật file `data/products_data.json`
2. Chạy lại `python load_data.py`
3. Restart API server

### Thay đổi chunking strategy
1. Sửa parameters trong `text_chunker.py`
2. Chạy lại script load data
3. Restart services

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch
3. Commit changes
4. Push to branch
5. Tạo Pull Request

## 📝 License

MIT License - xem file LICENSE để biết thêm chi tiết.

## 📞 Hỗ trợ

- 📧 Email: support@example.com
- 🐛 Issues: GitHub Issues
- 📖 Docs: API Documentation tại /docs

## 🎉 Changelog

### v2.0.0 (Current)
- ✨ Thêm Text Chunking
- 🚀 Cải thiện độ chính xác tìm kiếm
- 📊 Thêm analytics và metrics
- 🎨 Cải thiện UI/UX

### v1.0.0
- 🎯 Vector search cơ bản
- 🌐 FastAPI + Streamlit
- 📱 Responsive UI

---

**Được phát triển với ❤️ bằng Python, FastAPI, Streamlit và MongoDB Atlas**