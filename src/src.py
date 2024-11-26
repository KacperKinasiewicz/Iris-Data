import matplotlib.pyplot as plt
import numpy as np

# załadowanie danych z pliku -------------------------------------------------------------------------------------------
plik = "data1.csv"
dane = np.genfromtxt('data1.csv', delimiter=',')

dlugosc_dzialki = dane[:, 0].tolist()  # ":" - wiersze, "0" - z pierwszej kolumny
szerokosc_dzialki = dane[:, 1].tolist()
dlugosc_platka = dane[:, 2].tolist()
szerokosc_platka = dane[:, 3].tolist()
gatunek = dane[:, 4].tolist()

setosa = 0
versicolor = 1
virginica = 2

# 1.1 - liczebności poszczególnych gatunków ----------------------------------------------------------------------------
liczebnosc_setosa = gatunek.count(setosa)
liczebnosc_versicolor = gatunek.count(versicolor)
liczebnosc_virginica = gatunek.count(virginica)
liczebnosc_razem = len(gatunek)

udzial_setosa = liczebnosc_setosa / liczebnosc_razem * 100
udzial_versicolor = liczebnosc_versicolor / liczebnosc_razem * 100
udzial_virginica = liczebnosc_virginica / liczebnosc_razem * 100

# tabela 1
print("\nTabela 1. Liczności gatunków irysów")
print("Gatunek      Liczebność (%)")
print(f"Setosa       {liczebnosc_setosa} ({udzial_setosa:.1f}%)")
print(f"Versicolor   {liczebnosc_versicolor} ({udzial_versicolor:.1f}%)")
print(f"Virginica    {liczebnosc_virginica} ({udzial_virginica:.1f}%)")
print(f"Razem        {liczebnosc_razem} (100.0%)")

# 1.2 - miary rozkładu każdej cechy ------------------------------------------------------------------------------------
def miary(cecha, nazwa):
    minimum = min(cecha)
    maksimum = max(cecha)
    srednia_arytmetyczna = sum(cecha) / len(cecha)
    mediana = np.median(cecha)
    q1 = np.quantile(cecha, 0.25)
    q3 = np.quantile(cecha, 0.75)
    odchylenie_standardowe = np.std(cecha)

    print(f"{nazwa} (cm)\n"
          f"minimum: {minimum:.2f}\n"
          f"średnia arytmetyczna: {srednia_arytmetyczna:.2f} (± {odchylenie_standardowe:.2f})\n"
          f"mediana: {mediana:.2f} ({q1:.2f} - {q3:.2f})\n"
          f"maksimum: {maksimum:.2f}\n")

# tabela 2
print("\nTabela 2. Charakterystyka cech irysów")
miary(dlugosc_dzialki, "Długość działki kielicha")
miary(szerokosc_dzialki, "Szerokość działki kielicha")
miary(dlugosc_platka, "Długość płatka")
miary(szerokosc_platka, "Szerokość płatka")

# 2.1 - histogram dla każdej cechy -------------------------------------------------------------------------------------
def histogram(cecha, tytul, etykieta):
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 18})

    plt.title(tytul)
    plt.xlabel(etykieta + " (cm)")
    plt.ylabel("Liczebność")

    bin_edges = np.arange(np.floor(min(cecha)), np.ceil(max(cecha)) + 0.5, 0.5)
    plt.hist(cecha, bins=bin_edges, edgecolor='black')
    plt.xticks(bin_edges)
    plt.show()

histogram(dlugosc_dzialki, "Długość działki kielicha", "Długość")
histogram(szerokosc_dzialki, "Szerokość działki kielicha", "Szerokość")
histogram(dlugosc_platka, "Długość płatka", "Długość")
histogram(szerokosc_platka, "Szerokość płatka", "Szerokość")

# 2.2 - wykres pudełkowy dla każdej cechy ------------------------------------------------------------------------------
dane_setosa = dane[dane[:, 4] == 0]
dane_versicolor = dane[dane[:, 4] == 1]
dane_virginica = dane[dane[:, 4] == 2]

dane_dlugosc_dzialki = [dane_setosa[:, 0].tolist(), dane_versicolor[:, 0].tolist(), dane_virginica[:, 0].tolist(), ]
dane_szerokosc_dzialki = [dane_setosa[:, 1].tolist(), dane_versicolor[:, 1].tolist(), dane_virginica[:, 1].tolist(), ]
dane_dlugosc_platka = [dane_setosa[:, 2].tolist(), dane_versicolor[:, 2].tolist(), dane_virginica[:, 2].tolist(), ]
dane_szerokosc_platka = [dane_setosa[:, 3].tolist(), dane_versicolor[:, 3].tolist(), dane_virginica[:, 3].tolist(), ]

def pudelkowy(cecha, tytul, etykieta):
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 18})

    plt.title(tytul)
    plt.xlabel("Gatunek")
    plt.ylabel(etykieta + " (cm)")

    plt.boxplot(cecha, tick_labels=["setosa", "versicolor", "virginica"])
    plt.show()

pudelkowy(dane_dlugosc_dzialki, "Długość działki kielicha", "Długość")
pudelkowy(dane_szerokosc_dzialki, "Szerokość działki kielicha", "Szerokość")
pudelkowy(dane_dlugosc_platka, "Długość płatka", "Długość")
pudelkowy(dane_szerokosc_platka, "Szerokość płatka", "Szerokość")

# 3 - wykres punktowy z naniesioną linią regresji liniowej -----------------------------------------------------------
def punktowy(cecha1, cecha2, etykieta1, etykieta2):
    plt.figure(figsize=(6, 6))
    plt.rcParams.update({'font.size': 15})

    wspolczynnik_pearsona = np.corrcoef(cecha1, cecha2)[0, 1]
    wspolczynnik_a, wspolczynnik_b = np.polyfit(cecha1, cecha2, 1)
    tablica1 = np.array(cecha1)
    wartosc_linii = wspolczynnik_a * tablica1 + wspolczynnik_b

    plt.title(f"r =  {wspolczynnik_pearsona:.2f}; y = {wspolczynnik_a:.1f}x + {wspolczynnik_b:.1f}")
    plt.xlabel(etykieta1 + " (cm)")
    plt.ylabel(etykieta2 + " (cm)")
    
    plt.scatter(cecha1, cecha2)
    plt.plot(cecha1, wartosc_linii, color='red')
    plt.show()

punktowy(dlugosc_dzialki, szerokosc_dzialki, "Długość działki kielicha", "Szerokość działki kielicha")
punktowy(dlugosc_dzialki, dlugosc_platka, "Długość działki kielicha", "Długość płatka")
punktowy(dlugosc_dzialki, szerokosc_platka, "Długość działki kielicha", "Szerokość płatka")
punktowy(szerokosc_dzialki, dlugosc_platka, "Szerokość działki kielicha", "Długość płatka")
punktowy(szerokosc_dzialki, szerokosc_platka, "Szerokość działki kielicha", "Szerokość płatka")
punktowy(dlugosc_platka, szerokosc_platka, "Długość płatka", "Szerokość płatka")