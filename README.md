# Penjelasan Detil Algoritma Genetika - Knapsack Problem (main.ipynb)

File `main.ipynb` ini berisi implementasi utuh dari Algoritma Genetika (Genetic Algorithm / GA) dalam satu buah *Jupyter Notebook* untuk memecahkan persoalan optimasi *Knapsack Problem*. Tujuan dari algoritma ini adalah menentukan barang mana saja yang harus dimasukkan ke dalam ransel (tas) agar nilai/harganya maksimal namun total bobotnya tidak melebihi kapasitas tas.

Berikut adalah penjelasan dan bedah kode secara menyeluruh dan detail baris per baris.

---

## 1. Import *Library*
Blok pertama adalah pemanggilan (import) berbagai *library* standar Python yang akan digunakan.

```python
import random
import matplotlib.pyplot as plt
import numpy as np
```
- `import random`: Digunakan untuk menghasilkan angka acak. Angka acak sangat penting pada Algoritma Genetika untuk membangkitkan gen populasi awal, persilangan (crossover), dan mutasi.
- `import matplotlib.pyplot as plt`: Digunakan untuk membuat grafik (plot) riwayat perkembangan nilai fitness agar kinerja algoritma dapat diobservasi secara visual.
- `import numpy as np`: Library *NumPy* diimpor, yang umumnya menyediakan operasi array numerik secara efisien, meskipun pada *script* spesifik ini tidak banyak digunakan secara langsung.

---

## 2. Pembuatan Populasi Awal (Inisialisasi)
Pada GA, solusi disimulasikan sebagai "kromosom". Satu kromosom berisi himpunan "gen" (dalam kasus ini: array biner 0 dan 1). Nilai `1` artinya barang tersebut akan diambil, sedangkan `0` tidak.

```python
def inisialisasi_populasi(jumlah_populasi, jumlah_gen):
    populasi = [] # Mendeklarasikan list kosong untuk menampung seluruh kromosom
    
    # Melakukan perulangan untuk membuat kromosom sebanyak nilai jumlah_populasi
    for i in range(jumlah_populasi):
        # Membentuk 1 list kromosom yang berisi elemen angka acak (0 atau 1) sepanjang jumlah gen
        kromosom = [random.randint(0, 1) for _ in range(jumlah_gen)]
        
        # Kromosom yang telah dibuat dimasukkan ke dalam list populasi
        populasi.append(kromosom)
        
    # Mengembalikan populasi awal yang telah lengkap terbentuk
    return populasi
```

---

## 3. Fungsi Penghitung *Fitness* (Fungsi Tujuan)
Fungsi ini digunakan untuk menilai seberapa baik/optimal suatu kromosom menyelesaikan masalah. Nilai fitness diukur dari total nilai harga barang. Jika tas meluap (kelebihan bobot), maka fitness akan menjadi 0 (diberi penalti penuh).

```python
def hitung_fitness(kromosom, barang, kapasitas_tas):
    total_harga = 0 # Inisialisasi awal nilai harga barang = 0
    total_bobot = 0 # Inisialisasi awal nilai bobot/berat barang = 0
    
    # Melakukan perulangan untuk setiap gen dalam satu kromosom
    for i in range(len(kromosom)):
        # Mengecek apakah pada indeks barang ke-i, gen bernilai 1 (artinya barang tersebut diambil)
        if kromosom[i] == 1:
            total_harga += barang[i][1] # Menambahkan harga barang (indeks tuple ke-1) ke total_harga
            total_bobot += barang[i][2] # Menambahkan berat barang (indeks tuple ke-2) ke total_bobot
            
    # Mengecek apakah total berat keseluruhan telah melebihi batas kapasitas_tas
    if total_bobot > kapasitas_tas:
        return 0  # Jika lebih, diberi penalti (fitness = 0) karena ini adalah solusi cacat/tidak sah
    else:
        return total_harga # Jika tas tidak jebol, nilai fitness-nya murni diambil dari total_harga
```

---

## 4. Proses Seleksi Orang Tua (*Selection*)
Fungsi seleksi berguna untuk memilih orang tua (*parent*) yang akan diwariskan gen-nya kepada generasi selanjutnya. Algoritma ini memuat 2 jenis seleksi.

### a. *Roulette Wheel Selection*
```python
def roulette_wheel_selection(populasi, fitness_populasi):
    # Menjumlahkan total nilai fitness dari semua kromosom di populasi
    total_fitness = sum(fitness_populasi) 
    
    # Antisipasi kasus jika seluruh populasi memiliki fitness = 0 (contohnya kelebihan bobot semua)
    if total_fitness == 0:
        idx = random.randrange(len(populasi)) # Ambil sebuah angka acak sebagai indeks
        return populasi[idx], idx # Kembalikan parent secara acak
        
    # Menghitung probabilitas dari setiap individu: fitness individu dibagi dengan total seluruh fitness
    probabilitas = [fitness / total_fitness for fitness in fitness_populasi]
    
    kumulatif_prob = [] # Variabel list untuk jangkauan batas probabilitas kumulatif
    kumulatif = 0 # Menyimpan penjumlahan probabilitas sejauh ini
    
    # Melakukan iterasi dari probabilitas setiap kromosom
    for p in probabilitas:
        kumulatif += p # Menambahkan probabilitas individu ke total kumulatif berjalan
        kumulatif_prob.append(kumulatif) # Menambahkan batas kumulatif tersebut ke list (rentang berujung maksimal 1.0)
        
    r = random.random() # Mengundi/Membangkitkan satu nilai desimal acak (antara 0 hingga 1)
    
    # Mengecek di rentang batas manakah hasil undian `r` terjatuh
    for i, kum_prob in enumerate(kumulatif_prob):
        if r <= kum_prob: # Jika jarum roulette berhenti/lebih kecil sama dengan batas kromosom ke-i
            return populasi[i], i # Kembalikan kromosom ke-i sebagai pemenang seleksi beserta letak indeksnya
            
    # Sebagai jalur aman (fallback), kembalikan elemen kromosom yang terakhir bila tak ada nilai batas yang tertangkap di atas
    return populasi[-1], len(populasi)-1 
```

### b. *Tournament Selection*
```python
def tournament_selection(populasi, fitness_populasi, k=3):
    # k adalah ukuran grup pertarungan. Pastikan grup peserta (k) tidak lebih besar dari panjang populasi yang ada
    if len(populasi) < k:
        k = len(populasi)
        
    # Mengambil sejumlah k elemen/indeks acak tanpa repetisi dari list rentang jumlah populasi
    peserta_indices = random.sample(range(len(populasi)), k)
    
    # Menggabungkan data peserta ke dalam list baru berupa array tuple (Kromosom, Fitness, Indeks Awal)
    peserta = [(populasi[i], fitness_populasi[i], i) for i in peserta_indices]
    
    # Mengurutkan list peserta turnamen secara menurun/descending (reverse=True) berdasarkan nilai fitness (x[1])
    peserta.sort(key=lambda x: x[1], reverse=True)
    
    # Mengambil peserta urutan indeks [0] (juara 1 / fitness terbesar) untuk dikembalikan array gen & letak indeksnya
    return peserta[0][0], peserta[0][2]
```

---

## 5. Persilangan Gen (*Crossover*)
Proses persilangan ini meniru cara makhluk hidup mewariskan gen dari kedua orang tuanya dengan cara menukar sebagian informasi antar induk, menghasilkan varian keturunan baru.

### a. *One Point Crossover*
```python
def one_point_crossover(parent1, parent2):
    # Memilih 1 titik potong acak antara indeks 1 hingga 1 elemen sebelum elemen terakhir
    titik_potong = random.randint(1, len(parent1)-1)
    
    # Membentuk anak1 dari potongan parent1 (mulai kiri sampai titik potong) digabung parent2 (mulai titik potong sampai kanan)
    anak1 = parent1[:titik_potong] + parent2[titik_potong:]
    # Membentuk anak2 dari sisa kebalikannya (kiri milik parent2, kanan milik parent1)
    anak2 = parent2[:titik_potong] + parent1[titik_potong:]
    
    # Mengembalikan 2 anak keturunan tersebut
    return anak1, anak2
```

### b. *Two Point Crossover*
```python
def two_point_crossover(parent1, parent2):
    # Memilih titik potong pertama
    titik1 = random.randint(1, len(parent1)-2)
    # Memilih titik potong kedua yang indeks letaknya selalu sesudah titik potong pertama
    titik2 = random.randint(titik1+1, len(parent1)-1)
    
    # Menyematkan bagian tengah milik parent2 ke tengah parent1
    anak1 = parent1[:titik1] + parent2[titik1:titik2] + parent1[titik2:]
    # Menyematkan bagian tengah milik parent1 ke tengah parent2
    anak2 = parent2[:titik1] + parent1[titik1:titik2] + parent2[titik2:]
    
    # Mengembalikan anak hasil potongan dua titik
    return anak1, anak2
```

### c. *Uniform Crossover*
```python
def uniform_crossover(parent1, parent2):
    # Membangkitkan sebuah `mask` (bisa dianggap sebagai topeng cetak) berisikan bilangan acak 0 atau 1 sepanjang gen
    mask = [random.randint(0, 1) for _ in range(len(parent1))]
    anak1 = []
    anak2 = []
    
    # Melakukan iterasi di sepanjang setiap indeks gen kromosom parent
    for i in range(len(parent1)):
        # Jika nilai topeng di indeks tersebut bernilai 0 (tetap)
        if mask[i] == 0:
            anak1.append(parent1[i]) # anak1 mengambil nilai gen parent1
            anak2.append(parent2[i]) # anak2 mengambil nilai gen parent2
        # Jika bernilai 1 (bertukar / bersilang)
        else:
            anak1.append(parent2[i]) # anak1 mengambil gen dari milik parent2
            anak2.append(parent1[i]) # anak2 mengambil gen dari milik parent1
            
    return anak1, anak2
```

---

## 6. Mutasi Gen (*Mutation*)
Mutasi digunakan untuk mencegah algoritma mencapai *premature convergence* (konvergen terlalu dini/solusi tidak berkembang) dengan mengubah acak (flip) bagian kecil dari gen.

### a. *Swap Mutation*
```python
def swap_mutation(kromosom):
    kromosom = list(kromosom) # Memastikan objek dalam tipe data 'list'
    # Mengambil dua posisi secara acak dengan batas indeks panjang kromosom
    posisi1, posisi2 = random.sample(range(len(kromosom)), 2)
    
    # Menukar posisi isi dari dua elemen tersebut (Elemen Posisi 1 ditaruh di 2, begitupun sebaliknya)
    kromosom[posisi1], kromosom[posisi2] = kromosom[posisi2], kromosom[posisi1]
    return kromosom
```

### b. *Inversion Mutation*
```python
def inversion_mutation(kromosom):
    # Mengundi batas awal letak indeks yang akan disorot
    posisi1 = random.randint(0, len(kromosom) - 2)
    # Mengundi batas akhir rentang sorotan
    posisi2 = random.randint(posisi1 + 1, len(kromosom) - 1)
    
    # Membalik array elemen yang disorot tersebut dengan reverse(), dan menaruhnya kembali ke posisi semula
    kromosom[posisi1:posisi2] = list(reversed(kromosom[posisi1:posisi2]))
    return kromosom
```

### c. *Uniform Mutation*
```python
def uniform_mutation(kromosom, mutation_rate=0.1):
    kromosom = list(kromosom)
    # Looping membaca isi masing-masing gen dalam kromosom
    for i in range(len(kromosom)):
        # Jika peluang acak (0-1) yang terpilih jatuh di bawah probabilitas mutation_rate (misal 10%)
        if random.random() < mutation_rate:
            # Merubah bit 1 menjadi 0, atau bit 0 menjadi 1 menggunakan operasi matematis (1 - n)
            kromosom[i] = 1 - kromosom[i] 
    return kromosom
```

---

## 7. Eksekutor Program (Blok Kode Utama)

```python
# Daftar Barang: Diwakili sebagai tipe tuple (Nama, Harga, Bobot)
barang = [
    ("Barang1", 60, 10), ("Barang2", 100, 20), ("Barang3", 120, 30),
    ("Barang4", 90, 25), ("Barang5", 69, 11), ("Barang6", 70, 9),
    ("Barang7", 80, 15), ("Barang8", 90, 10), ("Barang9", 25, 3)
]

# Fungsi Controller / Penjalin siklus utama Algoritma
def run_ga(jumlah_generasi, jumlah_populasi, prob_crossover, prob_mutasi, kapasitas_tas):
    # Ukuran satu buah kromosom (banyaknya titik gen) mengacu pada banyaknya jumlah barang 
    jumlah_gen = len(barang)
    
    # Tahap 1: Membangun populasi secara teracak dengan memanggil fungsi di atas
    populasi = inisialisasi_populasi(jumlah_populasi, jumlah_gen)
    
    # Variabel array kosong sebagai buku catatan (log) riwayat performa GA (Berguna untuk diagram visual nanti)
    best_fitness_list = []
    worst_fitness_list = []
    avg_fitness_list = []
    all_fitness = []
    
    # Catatan tunggal rekam jejak kromosom (genetik) terbaik sepanjang generasi dihidupkan
    best_individu = None
    best_fitness_overall = 0
    
    # === MULAI SIKLUS UTAMA ===
    # Looping terus menerus hingga generasi mencapai batasan (jumlah_generasi)
    for generasi in range(jumlah_generasi):
        
        # Tahap 2: Menilai tingkat kebugaran setiap individu menggunakan list comprehension array
        fitness_populasi = [hitung_fitness(ind, barang, kapasitas_tas) for ind in populasi]
        
        # Mendapatkan nilai puncak atas, puncak terbawah, dan rata-rata performa dari generasi saat ini
        best_fitness = max(fitness_populasi)
        worst_fitness = min(fitness_populasi)
        avg_fitness = sum(fitness_populasi) / len(fitness_populasi)
        
        # Simpan nilai di atas ke buku rekam jejak
        best_fitness_list.append(best_fitness)
        worst_fitness_list.append(worst_fitness)
        avg_fitness_list.append(avg_fitness)
        all_fitness.append(fitness_populasi.copy()) # menyimpan keseluruhan sebaran 1 array populasi utuh
        
        # Jika nilai puncak (terbaik) generasi ini mengalahkan rekam jejak puncak global secara menyeluruh
        if best_fitness > best_fitness_overall:
            best_fitness_overall = best_fitness # Perbarui catatan fitness puncak overall
            index_best = fitness_populasi.index(best_fitness) # Ketahui di nomor indeks manakah sang jawara tersebut berada 
            best_individu = populasi[index_best] # Simpan rupa persis susunan array kromosom dari si jawara (best_individu)
            
        new_populasi = [] # Wadah kosong penampungan para keturunan (new generation)
        
        # === TAHAP EVOLUSI ===
        # Lakukan perulangan hingga wadah tempat generasi keturunan baru (new_populasi) penuh seperti kapasitas ukuran populasi saat ini
        while len(new_populasi) < jumlah_populasi:
            
            # Tahap 3: Pemilihan (Seleksi) Calon Induk Pertama
            # Panggil roulette_wheel_selection untuk mendelegasikan parent pertama beserta menangkap info indeksnya
            parent1, idx1 = roulette_wheel_selection(populasi, fitness_populasi)
            
            # Mendata list indeks tersisa pada populasi yang ada terkecuali (bukan) parent1
            available_indices = [i for i in range(len(populasi)) if i != idx1]
            
            # Memastikan ketersediaan populasi kandidat
            if not available_indices: # Jika sisa angka habis/populasi cuma berjumlah 1
                parent2 = parent1 # Biarkan parent2 sama seperti parent1 (Klone mandiri)
            else:
                # Bila ada sisa, lemparkan kembali turnamen seleksi roulette wheel untuk mendapatkan parent2 di populasi sisanya 
                parent2, _ = roulette_wheel_selection([populasi[i] for i in available_indices], [fitness_populasi[i] for i in available_indices])
            
            # Tahap 4: Mengawinkan / Menyilangkan kedua gen parent lewat Crossover (Persilangan)
            # Mengeksekusi peluang probabilitas dari sebuah angka dadu random dengan syarat pemicu (contoh prob_crossover = 0.5 atau 50% persilangan akan dijalankan)
            if random.random() < prob_crossover:
                anak1, anak2 = one_point_crossover(parent1, parent2) # Menghasilkan 2 keturunan yang disilang
            else:
                anak1, anak2 = parent1[:], parent2[:] # Menyalin utuh murni 100% gen parent kepada masing-masing anak (karena tak ada proses persilangan)
                
            # Tahap 5: Perubahan Minor (Mutasi)
            # Melempar dadu undian lagi, jika angka kebetulan rendah menyentuh batas prob_mutasi (ex: 0.1 / 10%), maka anak1 dipaksakan bermutasi dengan swap
            if random.random() < prob_mutasi:
                anak1 = swap_mutation(anak1)
                
            # Melakukan perlakuan dadu undian pe-mutasian mutlak secara individual juga kepada genetik anak2 
            if random.random() < prob_mutasi:
                anak2 = swap_mutation(anak2)
                
            # Menambah/memasukkan list yang berisi 2 orang anak tersebut ke dalam wadah `new_populasi`
            new_populasi.extend([anak1, anak2])
            
        # Jika dari perulangan genap di atas ukuran tempat tertumpah lebih batas panjang aslinya, maka pangkas berlebihannya hingga pas berjumlah konstan limit `jumlah_populasi` 
        populasi = new_populasi[:jumlah_populasi]

    # === TAHAP PENAMPILAN GRAFIK VISUAL ===
    # Membuat figure matplotlib baru agar lebih lebar dengan skala kanvas visual (Lebar 12, Tinggi 7) 
    plt.figure(figsize=(12, 7)) 
    
    # Melakukan perulangan dari dimensi generasi
    for i in range(jumlah_generasi):
        # Penentuan koordinat garis X yang konstan berada di urutan bilangan iterasi sumbu Generasinya 
        x = [i+1]*len(all_fitness[i])
        # Membaca isi sumbu array koordinat Y dari rekam jejak sebaran seluruh nilai populasi individu iterasi saat ini (all_fitness)
        y = all_fitness[i]
        
        # Mengecat taburan Scatter-Plot buram / alpha (bayangan pudar) rupa fitness para generasi secara sebaran abu-abu pudar
        plt.scatter(x, y, color='gray', alpha=0.1)
        
    # Membangun skema Garis Trend Biru tebal pada hasil rekaman kebugaran/Fitness Generasi Terbaik (best_fitness)
    plt.plot(range(1, jumlah_generasi+1), best_fitness_list, color='blue', label='Fitness Tertinggi')
    
    # Membangun skema Garis Trend Kuning pada hasil terekam Kebugaran individu Terlemah
    plt.plot(range(1, jumlah_generasi+1), worst_fitness_list, color='yellow', label='Fitness Terendah')
    
    # Membangun skema Garis Merah perwujudan Tren kebugaran Rata-Rata diantara anggota populasi generasi tersebut
    plt.plot(range(1, jumlah_generasi+1), avg_fitness_list, color='red', label='Fitness Rata-rata')
    
    # Set Judul Grafik
    plt.title('Perkembangan Nilai Fitness')
    
    # Set Label/Keterangan tulisan untuk sumbu horizontal (X)
    plt.xlabel('Generasi')
    
    # Set Label keterangan vertikal untuk sumbu ordinat Y
    plt.ylabel('Nilai Fitness')
    
    # Tampilkan box kunci indikator penunjuk warna identitas plot
    plt.legend()
    
    # Tampilkan mode petak jaring background (grid) agar penglihatan data mudah diproyeksikan
    plt.grid(True)
    
    # Perintahkan komputer mengeksekusi render jendela visual gambar Grafik di layar / output IPython (Jupyter Notebook)
    plt.show()

    # === PENERJEMAHAN KESIMPULAN / HASIL OPTIMAL KE MANUSIA ===
    # Mengonversi/membedah susunan Kromosom array biner Pemenang ke dalam representasi List Nama-Nama String Barang, dimana yang memiliki gen angka `1` adalah item barang yang sah Terambil 
    selected_items = [barang[i][0] for i in range(len(best_individu)) if best_individu[i] == 1]
    
    # Menghitung nominal ulang Harga Uang Murni / Fitness Final mutlak dari keranjang the best individu 
    selected_value = hitung_fitness(best_individu, barang, kapasitas_tas)
    
    # Merekap totalitas jumlah perhitungan dari indeks Berat/Bobot [indeks ke-2 elemen array Barang] (yang terambil/bertanda `1` pada best individu saja)
    selected_weight = sum([barang[i][2] for i in range(len(best_individu)) if best_individu[i] == 1])

    # Print Kesimpulan Ke Console Terbuka
    # Mencetak/Menulis Harga Total Ransel (Nilai Max)
    print(f"Nilai Fitness Terbaik: {selected_value}")
    
    # Mencetak peraihan beban kapasitas ransel akhir agar kita yakin terbukti muat tidak menembus daya tampung 
    print(f"Total Bobot: {selected_weight}")
    
    # Menampilkan tajuk/daftar
    print("Barang Terpilih:")
    
    # Melakukan enumerasi iterasi print poin nama barang yang telah dimasukkan list selected_items (Berdasarkan Kromosom Terbaik)
    for item in selected_items:
        print(f"- {item}")

# Menjalankan pemanggilan Fungsi Pusat utama Penggerak GA dengan mengirim argumen setelan metrik parameternya 
run_ga(
    jumlah_generasi=50,      # Pengaturan Algoritma akan menjalankan 50 siklus Generasi penuh (Tiap Generasi akan berisi fase seleksi + Crossover + Mutasi)
    jumlah_populasi=20,      # Mengatur jumlah populasi ketetapan per generasi adalah berjumlah 20 kromosom eksis 
    prob_crossover=0.5,      # Titik threshold probabilitas terjadinya aksi silang gen di titik 50%
    prob_mutasi=0.1,         # Titik Threshold terjadinya flip pada sebuah gen Mutasi (sebesar 10%)
    kapasitas_tas=50         # Parameter Batas maksimal / Limit muatan ransel kasus The Knapsack Problem ini adalah 50 
)
```
