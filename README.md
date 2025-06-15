Projekt na podstawy sztucznej inteligancji - Dorian Rząsa, Bartosz Zabdyr

Projekt polega na stworzeniu agenta sztucznej inteligencji uczącego się jazdy w środowisku `MountainCar-v0`, korzystając z algorytmu Proximal Policy Optimization (PPO). Celem agenta jest nauczenie się efektywnego manewrowania, aby dojechać do celu.


## 🧠 Algorytm: Proximal Policy Optimization (PPO)

## 📦 Zawartość repozytorium

- `ppo.py` – implementacja agenta PPO (sieci neuronowe, wybór akcji, aktualizacja wag)
- `train.py` – skrypt do trenowania agenta w środowisku `MountainCar-v0`
- `main.py` – skrypt odpalający prezentacje danego modelu

## 🛠️ Wymagania systemowe

Poniżej zamieszczono wymagane wersje bibliotek:

| Biblioteka     | Wersja         |
|----------------|----------------|
| `numpy`        | 2.2.5          |
| `torch`        | 2.7.0+cu126    |
| `pygame`       | 2.6.1          |
| `gymnasium`    | 1.1.1          |
|  Python        | 3.12.3         |

### Instalacja zależności

```bash
pip install numpy==2.2.5
pip install torch==2.7.0+cu126
pip install pygame==2.6.1
pip install gymnasium==1.1.1

