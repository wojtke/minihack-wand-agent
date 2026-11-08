# Plan: agent PPO + LSTM + RND/E3B

Plik roboczy gałęzi `ppo-agent`. **Usunąć osobnym commitem przed merge do `main`.**

## Zasady

- Jedna zmiana naraz. Każdy etap kończy się działającym agentem albo narzędziem.
- Kryteria rozdzielają dwie rzeczy:
  - **poprawność implementacji**: sprawdzana na zadaniach o znanym wyniku, gdzie porażka oznacza błąd w kodzie;
  - **wynik metody na WoD**: mierzony i zapisywany zawsze, ale bez progu do spełnienia, bo słaby wynik może być cechą metody, nie błędem.
- Każdy dodatek (LSTM, RND, E3B) włącza się i wyłącza w konfiguracji, żeby dało się go porównać i wykorzystać przy pytaniu badawczym.
- Wzorzec: `ppo_atari_lstm.py` z CleanRL (licencja MIT), z adnotacją w kodzie.

## Sprzęt

- Mac (M5): długie treningi, 8 środowisk, jeden trening naraz, kolejne seedy po kolei (np. w nocy, z `caffeinate`).
- PC: praca w ciągu dnia, bez treningów.
- Kod na gałęzi `ppo-agent`, przenoszony przez git.

## Etap 0: nauka (bez kodu)

Przeczytać `ppo_atari_lstm.py` (CleanRL) i „The 37 Implementation Details of PPO”.

- [ ] Potrafię opisać: zbieranie porcji, GAE, clipping, stan LSTM na początku porcji.

## Etap 1: wspólna infrastruktura

Równoległe środowiska (`wand_agent/vector.py`, 8 gier, po 2 na poziom), wspólny interfejs agenta `reset()` / `act(obs)` (`wand_agent/agents.py`), `evaluate.py` i `play.py` na tym interfejsie. Logi treningu przechodzą do etapu 2, gdzie są potrzebne.

- [x] Stare testy przechodzą.
- [x] Nowy `evaluate.py` daje dla checkpointu DQN 20/20/12/0 wygranych (20 gier na poziom od seeda 210000000, jak w `checkpoints/export-validation.json`; wszystkie średnie identyczne).
- [x] Ten sam seed daje te same gry w procesach i w jednym procesie (test).
- [x] Prędkość: losowy agent, 8 gier: ok. 15 tys. akcji/s (10 mln kroków ≈ 11 min gry). Mniej niż 32 tys./s z pomiaru bez synchronizacji, bo reset gry (ok. 1,2 ms) wstrzymuje pozostałe gry w tym samym kroku. Wystarcza; nie optymalizujemy.
- [x] Agent losowy, 1000 gier na poziom: Easy 6,2% (95%: 4,9–7,9%), pozostałe 0%.

## Etap 2: PPO bez LSTM (MLP na 463 cechach)

`wand_agent/ppo.py`, `configs/ppo.json`, `configs/ppo-cartpole.json`, `tests/test_ppo.py`. Aktor i krytyk jako osobne sieci 463 → 128 → 128 (przełącznik `shared` daje wspólny tułów).

Poprawność:
- [x] Test GAE zgodny z wynikiem policzonym ręcznie (gra z → f → l na Easy); koniec gry przerywa liczenie przewagi, ostatni krok porcji bierze ocenę krytyka.
- [x] `CartPole-v1`, 500 tys. kroków, seed 1: średnia z ostatnich 100 gier dochodzi do 490–498 (maks. 500), na końcu 476. Trwa ok. 20 s.
- [x] Logi w `metrics.jsonl`: entropia, KL, clipfrac, wyjaśniona wariancja, straty, learning rate, nagroda z porcji. Gry w `episodes.jsonl`.

Wynik na WoD (zapisany, bez progu), 1 seed:
- [x] Wszystkie poziomy, 3 mln kroków (`runs/ppo-wod-3m-s1`): Easy 100% po ok. 80 tys. kroków; Medium, Hard, Pro 0%. Zwrot na Pro rośnie z ok. −8 do ok. 0: agent uczy się tylko nie marnować ruchów. Na Medium mija różdżkę (nie nauczył się `,`, bo na Easy jest autopickup) i ginie od minotaura; na Hard ginie po ok. 18 krokach. Ewaluacja 200 gier na poziom: 200/0/0/0, tak samo z losowaniem akcji (`--sample`).
- [x] Samo Easy, 300 tys. kroków: 100% po ok. 40 tys. kroków.
- [x] Porównanie: agent losowy Easy 6,2%, reszta 0%; DQN (checkpoint) 100/99,8/80/0%. PPO bez dodatków jest na Medium i Hard dużo słabszy od DQN, który ma bonus +0,01 za nowe pole (tu `visit_bonus` = 0).
- [ ] Do sprawdzenia przy porównaniu: PPO z tym samym bonusem co DQN (`visit_bonus` 0,01), więcej seedów.

Prędkość treningu: ok. 4 tys. akcji/s zamiast 15 tys. z samej gry. Po nauczeniu Easy każda gra Easy trwa 3 kroki, więc prawie w każdym kroku któraś gra się restartuje (1,2 ms), a pozostałe czekają. Sieć to ok. 0,1 ms na krok, aktualizacja ok. 0,13 ms na krok. 10 mln kroków to ok. 40 min.

## Etap 3: LSTM

Poprawność:
- [ ] Pamięć zeruje się na końcu gry (test).
- [ ] Przed pierwszą aktualizacją sieć liczy podczas uczenia te same prawdopodobieństwa akcji co podczas zbierania (test).
- [ ] Zadanie na pamięć o znanym wyniku: sygnał pokazany na początku gry, decyzja kilka kroków później. LSTM rozwiązuje je prawie zawsze, MLP zgaduje (ok. 50%).

Wynik na WoD (zapisany, bez progu):
- [ ] Porównanie z etapem 2 na wszystkich poziomach.

Kamień milowy: „działający agent” na prezentację 3–4 grudnia.

## Etap 4: RND

Poprawność:
- [ ] Bonus maleje dla wielokrotnie pokazywanej tej samej obserwacji (test).
- [ ] Ewaluacja liczy tylko natywne nagrody (test).

Wynik na WoD (zapisany, bez progu):
- [ ] Trening z bonusem i bez, te same ustawienia i seedy; krzywe bonusu w logach.

## Etap 5: E3B (opcjonalnie)

Poprawność:
- [ ] Aktualizacja Shermana–Morrisona zgodna z odwrotnością liczoną wprost (test).
- [ ] Macierz zeruje się na początku każdej gry (test).

Wynik na WoD (zapisany, bez progu):
- [ ] Porównanie z RND na tych samych seedach.

## Etap 6: porównanie z DQN

- [ ] Co najmniej 3 seedy treningowe na konfigurację: DQN, PPO, PPO+LSTM, PPO+LSTM+RND.
- [ ] Ewaluacja: 1000 gier na poziom, seedy spoza treningu.
- [ ] Wykresy (średnia i przedział ufności) oraz tabela do raportu.

Liczba seedów zależy od czasu jednego treningu, zmierzonego w etapie 2.

## Przed merge do `main`

- [ ] Usunąć ten plik (`git rm experiments/PLAN-ppo.md`).
