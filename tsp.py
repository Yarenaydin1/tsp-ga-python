import random
import math
import matplotlib.pyplot as plt

# ==========================
# 1. Parametreler
# ==========================
NUM_CITIES = 6          # Şehir sayısı
POP_SIZE = 50           # Popülasyon büyüklüğü
NUM_GENERATIONS = 200   # Nesil sayısı
MUTATION_RATE = 0.2     # Mutasyon olasılığı


# ==========================
# 2. Şehirleri random üret
# ==========================
def generate_cities(num_cities):
    """
    num_cities kadar şehrin (x, y) koordinatlarını [0, 100] aralığında rastgele üretir.
    """
    cities = []
    for _ in range(num_cities):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        cities.append((x, y))
    return cities


def distance(a, b):
    """
    İki nokta arasındaki Öklid mesafesi
    """
    (x1, y1) = a
    (x2, y2) = b
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# ==========================
# 3. Kromozom & Amaç Fonksiyonu
# ==========================
# Kromozom = şehir index permütasyonu, örn: [0, 2, 1, 4, 5, 3]

def path_length(chromosome, cities):
    """
    Verilen kromozomun (rota) toplam yol uzunluğunu hesaplar.
    Son şehirden ilk şehre dönüşü de ekler (kapalı tur).
    """
    total = 0.0
    for i in range(len(chromosome) - 1):
        c1 = cities[chromosome[i]]
        c2 = cities[chromosome[i + 1]]
        total += distance(c1, c2)

    # Tura geri dön: son şehir → ilk şehir
    first_city = cities[chromosome[0]]
    last_city = cities[chromosome[-1]]
    total += distance(last_city, first_city)

    return total


def fitness(chromosome, cities):
    """
    Minimize etmek istediğimiz büyüklük yol uzunluğu.
    GA ise maximize ettiği için:
    fitness = 1 / (1 + yol_uzunluğu) kullanıyoruz.
    """
    L = path_length(chromosome, cities)
    return 1.0 / (1.0 + L)


# ==========================
# 4. GA Operatörleri
# ==========================

def create_initial_population(pop_size, num_cities):
    """
    Rastgele permütasyonlardan oluşan başlangıç popülasyonu oluşturur.
    """
    population = []
    base = list(range(num_cities))
    for _ in range(pop_size):
        chrom = base[:]
        random.shuffle(chrom)
        population.append(chrom)
    return population


def tournament_selection(population, cities, k=3):
    """
    Turnuva seçimi:
    Rastgele k birey seç, en yüksek fitness'lı olanı döndür.
    """
    selected = random.sample(population, k)
    selected.sort(key=lambda chrom: fitness(chrom, cities), reverse=True)
    return selected[0][:]  # en iyinin kopyası


def crossover(parent1, parent2):
    """
    Order Crossover (OX)
    Permütasyon kodlaması için uygun çaprazlama operatörü.
    """
    size = len(parent1)
    child = [None] * size

    # Rastgele iki kesim noktası seç
    start = random.randint(0, size - 2)
    end = random.randint(start + 1, size - 1)

    # parent1'den [start, end] aralığını al
    child[start:end + 1] = parent1[start:end + 1]

    # parent2'den eksik şehirleri sırayla doldur
    p2_idx = 0
    for i in range(size):
        if child[i] is None:
            while parent2[p2_idx] in child:
                p2_idx += 1
            child[i] = parent2[p2_idx]
            p2_idx += 1

    return child


def mutate(chromosome, mutation_rate=0.1):
    """
    Swap mutasyon:
    Belirli bir olasılıkla kromozomun iki geninin yerini değiştirir.
    """
    if random.random() < mutation_rate:
        i = random.randint(0, len(chromosome) - 1)
        j = random.randint(0, len(chromosome) - 1)
        chromosome[i], chromosome[j] = chromosome[j], chromosome[i]


# ==========================
# 5. Rota Çizimi (Görsel)
# ==========================
def plot_route(cities, best_chrom):
    """
    En iyi rotayı 2B düzlemde çizer.
    """
    # Koordinatları sıraya göre al
    x = [cities[i][0] for i in best_chrom]
    y = [cities[i][1] for i in best_chrom]

    # Tura geri dönüş için ilk şehri tekrar ekle
    x.append(cities[best_chrom[0]][0])
    y.append(cities[best_chrom[0]][1])

    plt.figure(figsize=(7, 7))

    # Rota çiz
    plt.plot(x, y, marker="o", linestyle="-", linewidth=2)

    # Şehir isimleri (S1, S2, ...)
    for idx in range(len(best_chrom)):
        cx, cy = cities[best_chrom[idx]]
        plt.text(cx + 1, cy + 1, f"S{best_chrom[idx] + 1}", fontsize=12)

    plt.title("Genetic Algorithm – TSP Rota Çizimi")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


# ==========================
# 6. Ana GA Döngüsü
# ==========================
def genetic_algorithm():
    # Şehirleri üret (her run'da random)
    cities = generate_cities(NUM_CITIES)

    # Şehirleri S1, S2, S3... şeklinde yazdır
    print("Şehir koordinatları (S_i = (x, y)):")
    for i, (x, y) in enumerate(cities):
        print(f"S{i + 1}: ({x:.2f}, {y:.2f})")
    print("-" * 40)

    # Başlangıç popülasyonu
    population = create_initial_population(POP_SIZE, NUM_CITIES)

    best_chrom = None
    best_fit = -1.0

    for gen in range(NUM_GENERATIONS):
        new_population = []

        # Yeni popülasyonu üret
        for _ in range(POP_SIZE):
            # Seçim
            parent1 = tournament_selection(population, cities)
            parent2 = tournament_selection(population, cities)

            # Çaprazlama
            child = crossover(parent1, parent2)

            # Mutasyon
            mutate(child, MUTATION_RATE)

            new_population.append(child)

        population = new_population

        # Neslin en iyisini bul
        for chrom in population:
            f = fitness(chrom, cities)
            if f > best_fit:
                best_fit = f
                best_chrom = chrom[:]

        # Her 20 nesilde bir durumu yazdır
        if (gen + 1) % 20 == 0:
            L = path_length(best_chrom, cities)
            print(f"Nesil {gen + 1:3d} | En iyi yol uzunluğu: {L:.3f}")

    # ==========================
    # SONUÇ
    # ==========================
    print("\n=== SONUÇ ===")
    print("En iyi rota (indexlerle):", best_chrom)

    # Index 0 → S1, index 1 → S2 ...
    rota = " → ".join(f"S{idx + 1}" for idx in best_chrom)
    rota += f" → S{best_chrom[0] + 1}"   # tura geri dön
    print("En iyi rota (şehir isimleri):", rota)

    en_iyi_yol = path_length(best_chrom, cities)
    print("Toplam yol uzunluğu:", f"{en_iyi_yol:.3f}")

    # ROTA ÇİZ
    plot_route(cities, best_chrom)


# ==========================
# 7. Program Girişi
# ==========================
if __name__ == "__main__":
    genetic_algorithm()
