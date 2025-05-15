FROM python:3.10-slim

# 필요한 시스템 패키지 설치
RUN apt update && apt install -y \
    git \
    wget \
    ffmpeg \
    libglfw3 \
    libgl1 \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libosmesa6-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 소스 코드 복사
COPY . .

# Python 패키지 설치
RUN pip install --upgrade pip && \
    pip install PyOpenGL==3.1.6 PyOpenGL_accelerate && \
    pip install -r requirements.txt

# 모델 및 기타 자산 다운로드
RUN python scripts/download_assets.py

# 컨테이너 시작 시 Flask 서버 실행
CMD ["python", "scripts/web/run_flask_server.py"]
