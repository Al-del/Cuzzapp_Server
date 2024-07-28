## 📚 Documentație Cuzzapp_Server


#### 📝 Descriere Generală
Acest proiect utilizează două modele de AI dezvoltate cu PyTorch pentru detectarea și recunoașterea textului în imagini. Aplicația este implementată folosind Flask și este containerizată cu Docker.

#### 🗂️ Structura Proiectului
* **app.py**: Fișierul principal al aplicației Flask.
* **src/test.py**: Conține funcțiile pentru predicția modelelor AI.
* **requirements.txt**: Lista de dependențe Python necesare pentru proiect.
* **Dockerfile**: Configurația pentru containerizarea aplicației.

#### 💻 Instalare și Configurare

1. **Clonarea Repozitoriului** 📂
  ```bash
  git clone <URL-repozitoriu>
  cd <nume-repozitoriu>
  ```

2. *Construirea Imaginilor Docker*🧱
   ```bash
   docker build -t ai-text-detection .
   ```

3. *Pornirea Containerului*🚀
   ```bash
   docker run -p 5000:5000 ai-text-detection
   ```

#### 📊Utilizare

1. *Endpoint-ul de Upload*📂
   - *URL*: /upload
   - *Metodă*: POST
   - *Descriere*: Permite încărcarea unei imagini pentru detectarea și recunoașterea textului.
   - *Exemplu de Request*:
     

2. *Endpoint-ul de Căutare*🔍
   - *URL*: /search
   - *Metodă*: GET
   - *Descriere*: Permite căutarea unei cărți după titlu în biblioteca Genesis.
   - *Exemplu de Request*:
    ```bash
     curl -X GET 'http://localhost:5000/search?q=book_title'
     ```

#### Detalii Tehnice

1. *Fișierul app.py*[

   - Inițializează aplicația Flask.

   - Definește endpoint-urile /upload și /search.
   - Utilizează funcția model_predicts din src/test.py pentru a procesa imaginile încărcate.

2. *Fișierul src/test.py*🤖
   - Conține funcțiile pentru încărcarea modelelor AI și efectuarea predicțiilor.
   - Utilizează PyTorch pentru inferență și OpenCV pentru procesarea imaginilor. 


3. *Fișierul requirements.txt*📝
   - Listează toate pachetele Python necesare pentru rularea aplicației.

4. *Fișierul Dockerfile*🚀
   - Definește pașii pentru construirea imaginii Docker, inclusiv instalarea dependențelor și copierea codului sursă.

#### 📦Dependențe
- Python 3.12
- PyTorch
- Flask
- OpenCV
- PIL
- NumPy
- Libgen API

[![My Skills](https://skillicons.dev/icons?i=python,pytorch,flask,opencv)](https://skillpythonicons.dev)  

#### Ghid instalare⬇️
git clone https://github.com/Al-del/Cuzzapp_Server.git
cd Cuzzapp_Server
docker build -t ai-text-detection .
docker run --gpus=all  -p 5000:5000 ai-text-detection
docker run --net=host -it -e NGROK_AUTHTOKEN=2hYSKegdUZVOvg3cdBVaAClL2bn_4AfN7YcUy6sN6NycqiWRq ngrok/ngrok:latest http --domain=reliably-expert-mammoth.ngrok-free.app 80
