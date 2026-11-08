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

Równoległe środowiska (8 gier, po 2 na poziom), wspólny interfejs agenta `reset()` / `act(obs)`, logi do plików.

- [ ] Stare testy przechodzą.
- [ ] Nowy `evaluate.py` daje dla checkpointu DQN 20/20/12/0 wygranych (20 gier na poziom od seeda 210000000, jak w `checkpoints/export-validation.json`).
- [ ] Ten sam seed daje te same gry w środowiskach równoległych.
- [ ] Losowy agent na 8 procesach: ok. 25–30 tys. akcji/s.

## Etap 2: PPO bez LSTM (MLP na 463 cechach)

`wand_agent/ppo.py`, `configs/ppo.json`, testy.

Poprawność:
- [ ] Test GAE zgodny z wynikiem policzonym ręcznie; koniec gry przerywa liczenie przewagi.
- [ ] Na `CartPole-v1` (gymnasium) średni wynik dochodzi do ok. 500, jak w publikowanych wynikach CleanRL. Porażka = błąd w kodzie.
- [ ] Logi: entropia, KL, clipfrac, wyjaśniona wariancja krytyka, straty, nagrody.

Wynik na WoD (zapisany, bez progu):
- [ ] Trening na samym Easy i na wszystkich czterech poziomach: krzywe uczenia i ewaluacja.
- [ ] Porównanie z losowym agentem (Easy 5,6% wygranych) i z DQN.

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
