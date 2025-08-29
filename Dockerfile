FROM python:3.10-slim

# Systempakete installieren, inkl. bash & dos2unix für Ghidra-Kompatibilität
RUN apt update && \
    apt install -y wget curl unzip bash dos2unix \
                   gcc g++ make build-essential \
                   binutils-arm-linux-gnueabi \
                   binutils-mips-linux-gnu \
                   python3-dev ca-certificates && \
    apt clean

# OpenJDK 21 (Temurin) manuell installieren
RUN mkdir -p /usr/lib/jvm && \
    curl -L https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.1%2B12/OpenJDK21U-jdk_x64_linux_hotspot_21.0.1_12.tar.gz \
    | tar xz -C /usr/lib/jvm && \
    ln -s /usr/lib/jvm/jdk-21.0.1+12 /usr/lib/jvm/java-21-openjdk-amd64

# JDK-Umgebung setzen
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Ghidra herunterladen & korrekt entpacken
WORKDIR /opt
RUN wget https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.3.2_build/ghidra_11.3.2_PUBLIC_20250415.zip && \
    unzip ghidra_11.3.2_PUBLIC_20250415.zip && \
    rm ghidra_11.3.2_PUBLIC_20250415.zip && \
    apt update && apt install -y dos2unix && \
    chmod +x /opt/ghidra_11.3.2_PUBLIC/support/analyzeHeadless && \
    dos2unix /opt/ghidra_11.3.2_PUBLIC/support/analyzeHeadless

# Ghidra-Umgebungsvariablen setzen
ENV GHIDRA_HOME=/opt/ghidra_11.3.2_PUBLIC
ENV PATH="${GHIDRA_HOME}/support:$PATH"

# JDK explizit für Ghidra setzen (verhindert Nachfragen im Headless-Modus)
RUN echo "VMPath=${JAVA_HOME}" > ${GHIDRA_HOME}/support/launch.properties

# Arbeitsverzeichnis im Container
WORKDIR /analyse_binary_llm

# Python-Abhängigkeiten installieren
RUN pip install --upgrade pip && \
    pip install torch \
                scikit-learn==1.1 \
                capstone \
                unicorn \
                numpy==1.23 \
                cython \
                wheel \
                setuptools \
                matplotlib \
                requests \
                httpx \
                openai \
                dspy-ai

# Optional: Ghidra-Skripte und Python-Code ins Image kopieren
# COPY ./ghidra_scripts /analyse_binary_llm/ghidra_scripts
# COPY ./load_cfile.py /analyse_binary_llm/

# Default Command (wenn du willst)
# CMD ["python3", "load_cfile.py"]

