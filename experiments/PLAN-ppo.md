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
- [x] PPO z tym samym bonusem co DQN (`visit_bonus` 0,01), 3 mln kroków (`runs/ppo-wod-3m-bonus-s1`): nadal Easy 100%, reszta 0%. Bonus jest zbierany (średnio Medium 0,21, Hard 0,15, Pro 0,52 na grę), ale entropia spada do ok. 0,07 i agent na Medium ginie jeszcze szybciej (średnio 22 kroki). Hipoteza: Easy daje prawie cały dodatni sygnał (35 tys. gier na 5,7 tys. gier Medium w ostatnich 500 tys. kroków), polityka szybko staje się prawie deterministyczna i przestaje eksplorować; DQN zachowuje 10% losowych akcji do końca.
- [ ] Do sprawdzenia: większy `ent_coef`, mniejszy udział Easy, więcej seedów.

Prędkość treningu: ok. 4 tys. akcji/s zamiast 15 tys. z samej gry. Po nauczeniu Easy każda gra Easy trwa 3 kroki, więc prawie w każdym kroku któraś gra się restartuje (1,2 ms), a pozostałe czekają. Sieć to ok. 0,1 ms na krok, aktualizacja ok. 0,13 ms na krok. 10 mln kroków to ok. 40 min.

## Etap 3: LSTM

`--lstm N` (lub `"lstm": N` w konfiguracji): wspólny tułów MLP → LSTM z N jednostkami → aktor i krytyk, jak w CleanRL `ppo_atari_lstm.py`. Zadanie kontrolne w `wand_agent/toy.py`, ustawienia w `configs/ppo-memory.json`.

Poprawność:
- [x] Pamięć zeruje się na początku nowej gry, tylko w tej jednej grze (test).
- [x] Przeliczenie całej porcji (także minibatcha z kilku gier) daje te same prawdopodobieństwa co gra krok po kroku (test); agent z checkpointu gra tak samo po `reset()`.
- [x] Zadanie na pamięć (sygnał w 1. kroku, decyzja po 5 pustych krokach), 200 tys. kroków, seed 1: bez pamięci 45–49% wygranych, z LSTM 32 99% po 25 tys. kroków, potem 100%.

Koszt: z LSTM 128 aktualizacja trwa ok. 260 ms na iterację zamiast 18 ms (pętla po 128 krokach przy przeliczaniu pamięci), czyli ok. 2 ms na krok wszystkich gier. Spodziewane ok. 2–2,5 tys. akcji/s na WoD.

Wynik na WoD (zapisany, bez progu), 1 seed:
- [x] PPO + LSTM 128, wszystkie poziomy, 3 mln kroków, bez bonusu (`runs/ppo-wod-3m-lstm128-s1`): tak samo jak bez pamięci, Easy 100%, Medium/Hard/Pro 0% (ewaluacja 200 gier na poziom: 200/0/0/0). Entropia zostaje wyższa (ok. 0,35 zamiast 0,07–0,5), Medium kończy się częściej limitem akcji niż śmiercią. Prędkość ok. 2,2 tys. akcji/s (23 min na 3 mln kroków).
- Wniosek na teraz: pamięć sama nie rozwiązuje problemu. Na Medium i Hard brakuje eksploracji (agent nigdy nie trafia na nagrodę), więc następny krok to etap 4 (RND) i/lub entropia oraz udział Easy (patrz etap 2).

Kamień milowy: „działający agent” na prezentację 3–4 grudnia.

## Etap 4: RND

`--rnd` (`wand_agent/rnd.py`): sieć docelowa (stała, losowa) i sieć ucząca się ją przewidywać; błąd przewidywania to bonus za nowość. Druga głowa krytyka uczy się bonusu, przewaga = 2 · przewaga z nagrody gry + 1 · przewaga z bonusu (jak w CleanRL `ppo_rnd_envpool.py`). Bonus nie zatrzymuje się na końcu gry. Obserwacje normalizowane bieżącą średnią i odchyleniem, bonus dzielony przez odchylenie swojej zdyskontowanej sumy.

Od tego etapu domyślny zestaw gier to 1× Easy i po 2× Medium, Hard, Pro (7 gier). Easy ma stałą mapę i rozwiązanie z 3 klawiszy, więc 2 gry Easy tylko zabierały kroki i dawały większość dodatniego sygnału. Wyniki z etapów 2–3 (po 2× Easy) nie są wprost porównywalne; dla porównania jest przebieg PPO bez RND na nowym zestawie.

Poprawność:
- [x] Bonus maleje dla wielokrotnie pokazywanej obserwacji (po treningu predyktora < 20% początkowego), a dla nowych zostaje wyższy; sieć docelowa się nie zmienia (test).
- [x] Agent z checkpointu RND gra tylko polityką; ewaluacja liczy tylko nagrody gry (test, `evaluate.py` bez bonusu).
- [x] Krótki trening z `--rnd --lstm` loguje bonus (surowy, znormalizowany, dla każdej gry) i stratę predyktora (test).

Wynik na WoD (zapisany, bez progu):
- [x] PPO bez RND, nowy zestaw gier, 3 mln kroków, seed 1 (`runs/ppo-wod-3m-mix1easy-s1`): Easy 100% po ok. 160 tys. kroków, Medium/Hard/Pro 0%; ewaluacja 200 gier na poziom: 200/0/0/0. Medium kończy się głównie limitem akcji, Hard śmiercią po ok. 20 krokach.
- [x] PPO + RND, 3 mln kroków, seed 1 (`runs/ppo-wod-3m-rnd-s1`, powtórzony identycznie jako `-s1b`) i seed 2 (`runs/ppo-wod-3m-rnd-s2`). Ewaluacja 200 gier: seed 1 0/0/0/0, seed 2 200/0/0/0.
  - Seed 1: agent w ciągu pierwszych 20 tys. kroków przestaje wygrywać Easy i do końca ginie tam po ok. 12 krokach. To nie jest błąd w RND: ten sam zestaw na samym Easy uczy się do 100%, a przy minimalnie innej konfiguracji (inna długość treningu, czyli inne wygaszanie LR) seed 1 uczy Easy normalnie. Trening jest powtarzalny przy identycznej komendzie, ale wrażliwy na drobne zmiany.
  - Bonus (znormalizowany) jest mały, ok. 0,001–0,01 na krok; najwyższy na Medium i Pro, prawie zero na Easy.
  - Na Medium, Hard i Pro RND nic nie daje: 0% wygranych, Hard ginie po ok. 25 krokach, Medium częściej ginie niż bez RND.
- Wniosek: sam RND na tych cechach nie wystarcza; do sprawdzenia większy `rnd_int_coef`, entropia, RND z LSTM, E3B (bonus liczony w obrębie gry).

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
