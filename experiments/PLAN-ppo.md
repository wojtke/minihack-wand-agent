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
- [x] Samo Medium (7 gier), 2 mln kroków, seed 1. W mieszanych treningach Medium/Hard były wygrane 0–2 razy na 3 mln kroków, za rzadko do nauki.
  - PPO (`runs/ppo-medium-2m-s1`): entropia spada do 0 w ciągu 18 tys. kroków, agent ginie w każdej grze, 0 wygranych w treningu.
  - PPO + bonus za nowe pole 0,01 (`runs/ppo-medium-2m-bonus-s1`): to samo, 1 wygrana w całym treningu.
  - PPO + RND (`runs/ppo-medium-2m-rnd-s1`): pierwsza wygrana po 14 tys. kroków, 0% do ok. 440 tys., potem w 80 tys. kroków skok do 96%, dalej 99–100%. Ewaluacja 200 gier: Easy 78, Medium 200, Hard 1, Pro 0 (Easy bez treningu na Easy).
- [x] Samo Hard + RND (7 gier), 3 mln kroków, seed 1 (`runs/ppo-hard-3m-rnd-s1`): 0 wygranych w 119 tys. gier, każda kończy się śmiercią po ok. 27 krokach; ewaluacja 0/0/0/0. Różdżka leży 3 pola od startu, ale agent do niej nie idzie: kręci się po pokoju, rzuca kamieniami (`f` strzela krzemieniem z kołczanu jaskiniowca, co daje nowe obserwacje i bonus RND), wchodzi w korytarz i ginie od minotaura. Model z samego Medium wygrywa Hard 1/200.
- [x] Curriculum Medium → Hard: start z `runs/ppo-medium-2m-rnd-s1/latest.npz` (`--init-from`, nowy optymalizator i nowe RND), samo Hard + RND, 3 mln kroków (`runs/ppo-medium-to-hard-3m-rnd-s1`): 10 wygranych na 113 tys. gier (pierwsza po 18 tys. kroków), poziom nie rośnie; ewaluacja 200 gier: Easy 71, Medium 0, Hard 0, Pro 0. Agent zapomniał Medium (było 200/200). Na starcie polityka z Medium na Hard traci dużo na karach (zwrot −2,1), więc pierwsze aktualizacje mocno ją zmieniają.
  - Różnica względem DQN (80% na Hard): DQN trzyma rzadkie wygrane w pamięci powtórek i uczy się z nich wiele razy; PPO widzi każdą wygraną w jednej aktualizacji i ją wyrzuca. Kandydaci: self-imitation learning (powtarzanie własnych udanych gier), Medium w zestawie przy treningu Hard (przeciw zapominaniu).
- [x] DQN od zera, domyślna konfiguracja (bonus 0,01 za nowe pole), 3 mln kroków, seed 1 (`runs/dqn-3m-s1`, ok. 14 min): ewaluacja 200 gier 200/0/0/0. W treningu 22 wygrane Medium na ok. 7,7 tys. gier, Hard i Pro nigdy. Wynik checkpointu DQN (100/99,8/80/0) pochodzi z wcześniejszego wieloetapowego treningu, nie z samej metody.
- [x] Kształtowanie nagrody tylko w treningu (`--wand-bonus`, `--death-penalty`, `wand_agent/env.py`, ewaluacja bez zmian): +0,5 gdy różdżka trafia do ekwipunku (−0,5 gdy znika; potencjał „ma różdżkę”, Ng i in. 1999), −1 za śmierć. Domyślny zestaw 7 gier, 3 mln kroków, seed 1.
  - PPO + RND + kształtowanie (`runs/ppo-wod-3m-rnd-shaped-s1`): Easy 100% po ok. 0,5 mln kroków; do 2,2 mln 0% na Medium i Hard, potem w ok. 100 tys. kroków Medium 81% → 98%, Hard 10% → ok. 60%. Ewaluacja 200 gier: **200/200/142/0** (Hard 71%), czyli poziom checkpointu DQN. Pod koniec trening nadal się poprawia.
  - PPO + kształtowanie bez RND (`runs/ppo-wod-3m-shaped-s1`): podnosi różdżkę na Medium (bonus 0,5 w każdej grze), ale nie wygrywa; Medium kończy się limitem akcji, entropia spada do 0,02. Ewaluacja 200/0/0/0.
  - PPO + RND + kształtowanie + LSTM 128 (`runs/ppo-wod-3m-rnd-shaped-lstm128-s1`, ok. 2,3 tys. akcji/s, 22 min): ewaluacja 200/0/0/0. Easy dopiero po ok. 1 mln kroków (bez LSTM 0,5 mln); na Medium podnosi różdżkę i dochodzi do limitu akcji, na Hard często podnosi różdżkę i ginie. Bez LSTM przełom przyszedł po 2,2 mln kroków, więc 3 mln może być za mało dla wolniej uczącego się LSTM; 1 seed.
  - Powtórzenia PPO + RND + kształtowanie, 3 mln kroków: seed 2 (`runs/ppo-wod-3m-rnd-shaped-s2`) ewaluacja **0/0/0/0**: agent przestaje strzelać już przed 0,3 mln kroków i kończy prawie każdą grę limitem akcji (także Easy, gdzie zły kierunek strzału zabija). Kara −1 za śmierć sprawia, że unikanie ryzyka jest lokalnym optimum. Seed 3 (`runs/ppo-wod-3m-rnd-shaped-s3`) ewaluacja **200/200/161/0**; Medium 91% już po 0,6 mln kroków. Czyli 2 z 3 seedów działają, wynik zależy od losu na początku treningu.
  - PPO + RND + kształtowanie, 10 mln kroków, seed 1 (`runs/ppo-wod-10m-rnd-shaped-s1`, 36 min): ewaluacja **200/200/169/0** (Hard 84,5%, przegrane Hard to limit akcji, nie śmierć). Medium 98% po 1 mln kroków, Hard rośnie do ok. 80% przy 2 mln i dalej tylko powoli (83–85%). Krótszy harmonogram LR daje inny przebieg niż run 3 mln z tym samym seedem.
  - PPO + RND + kształtowanie + LSTM 128, 10 mln kroków, seed 1 (`runs/ppo-wod-10m-rnd-shaped-lstm128-s1`, 77 min): Easy 100% po 1 mln, Medium i Hard 0% przez cały trening; na Medium podnosi różdżkę i dochodzi do limitu akcji. Ewaluacja **200/0/0/0**. LSTM nie pomaga przy tych cechach i budżecie; 1 seed.
  - Kara za śmierć, PPO + RND + bonus za różdżkę 0,5, 3 mln kroków, seedy 1–3 (Hard / 200):
    - kara 1,0 (`runs/ppo-wod-3m-rnd-shaped-s{1,2,3}`): 142 / 0 / 161, seed 2 przestaje strzelać i kończy limitem akcji;
    - kara 0 (`runs/ppo-wod-3m-rnd-wand-nodeath-s{1,2,3}`): 0 (Easy: natychmiastowa śmierć, natywnie tańsza niż czekanie) / 155 / 0 (Medium 200, Hard zawsze śmierć);
    - kara 0,25 (`runs/ppo-wod-3m-rnd-wand-death025-s{1,2,3}`): 169 / 167 / 0 (seed 3: Medium limit akcji, Hard śmierć).
    - Wniosek: kara zmienia, które seedy się psują, ale żadna wartość nie daje 3/3. Bez kary agent uczy się szybkiej śmierci, z dużą karą unika ryzyka.
  - Nowe opcje (kara 1,0, RND, bonus za różdżkę, seedy 1–3, Hard / 200; „M90%” = krok, w którym Medium przekracza 90% w treningu):
    - makro-akcje `--macro` (17 akcji: 8 ruchów, podniesienie, 8 × „zap różdżką w kierunku d” = z, f, d wewnątrz wrappera; każde natywne wejście liczy się do limitu akcji, więc ewaluacja jest na tej samej grze): **169 / 169 / 169**, M90% po 0,5–0,84 mln. Jedyna zmiana, która działa na 3/3 seedach. Kroki w configu liczą decyzje agenta, więc trening trwa ok. 2× dłużej (20 min).
    - kształtowanie „w linii z potworem” `--align-bonus 0.2` (potencjał Φ = 0,2, gdy agent ma różdżkę i widzi potwora w tym samym wierszu, kolumnie lub na przekątnej; F = γΦ(s') − Φ(s), Φ = 0 po końcu gry): 166 / 0 / 102. Nie naprawia seeda 2, seed 3 gorszy (98 gier Hard limitem akcji).
    - SIL `--sil` (Oh i in. 2018, tylko wygrane gry, osobny bufor na poziom, by tysiące krótkich wygranych Easy nie wypierały Medium; 4 × 512 próbek po każdej aktualizacji): 168 / 165 / 140. Naprawia seed 2, nie przyspiesza wyraźnie (M90% 0,72–2,35 mln).
    - 14 gier zamiast 7 (dwa razy ten sam zestaw): 168 / 169 / 158. Też naprawia seed 2.
    - Bez zmian w kodzie było już: osobna głowa wartości dla RND, normalizacja bonusu, nieepizodyczny zwrot wewnętrzny, γ = 0,99, λ = 0,95, entropia 0,01. Curriculum Medium → Hard sprawdzone wcześniej (zapomina Medium). Odwrotne curriculum wymagałoby własnych map; demonstracje / BC nie sprawdzone (najprostszy demonstrator to checkpoint DQN, który jednak też przegrywa wszystkie gry opisane niżej).
  - Kombinacje (kolejka `tools/run_queue.sh`; baza: RND + różdżka 0,5 + kara 1,0 + `--macro`; Hard / 200, w nawiasie M90% i H50% = krok, w którym Medium przekracza 90%, a Hard 50% w treningu):
    - samo makro (dla porównania): 169 / 169 / 169 (M90% 0,74 / 0,49 / 0,84 mln; H50% 0,83 / 0,59 / 1,01 mln), ok. 20 min;
    - makro + SIL (`runs/combo-macro-sil-s{1,2,3}`): **168 / 166 / 169**, M90% **0,19 / 0,18** / 1,55 mln, H50% **0,46 / 0,35** / 0,96 mln, ok. 21 min. 3/3 seedy działają, na 2 z 3 Medium i Hard uczą się 2–4× szybciej niż z samym makro. Wynik końcowy się nie poprawia, bo sufit to 169;
    - makro + 14 gier (`runs/combo-macro-14g-s{1,2,3}`): 167 / 168 / **0**, ok. 12,5 min. Seed 3 nigdy nie uczy się Medium (każda gra limitem akcji) ani Hard (8 wygranych w treningu);
    - makro + SIL + 14 gier (`runs/combo-macro-sil-14g-s{1,2,3}`): 166 / **19** / 169, ok. 13 min. Seed 2 jak wyżej: Medium 0% przez cały trening, Hard rośnie dopiero od 1,5 mln (45% na końcu, ewaluacja 19). Przy makro 14 gier psuje 2 z 6 runów, więc lepiej zostać przy 7;
    - 10 mln kroków, seed 1: makro + SIL (`runs/combo10m-macro-sil-s1`, 96 min) **169**, M90% 0,53 mln; makro + SIL + 14 gier (`runs/combo10m-macro-sil-14g-s1`, 55 min) **169**. Dłuższy trening nic nie dodaje ponad 169;
    - samo Pro (7× Pro) + makro + SIL, 10 mln kroków (`runs/pro-only10m-macro-sil-s1`, 17 min): **0 wygranych w ok. 10,6 tys. gier treningu**, ewaluacja Pro 0/200 (172 limit akcji, 28 śmierci). Pro bez nowego pomysłu (pamięć mapy, eksploracja całego labiryntu) się nie uczy.
    - Wniosek: najlepsza konfiguracja to **RND + różdżka 0,5 + kara 1,0 + makro + SIL na 7 grach**: 3/3 seedy, 166–169/169 wygrywalnych Hard i najszybsza nauka.
  - Sufit 169/200 na Hard to **błąd poziomu w MiniHack**, nie agenta: Hard losuje miejsce różdżki w `fillrect (1,1,5,5)` (`minihack/envs/skills_wod.py`), a wiersz 5 mapy to dolna ściana `|-----|`. W ok. 1/6 gier różdżka leży w ścianie (31 z 200 seedów ewaluacji). Nie da się jej podnieść („It's a wall.” z każdej strony), a bez niej minotaura nie da się zabić (checkpoint DQN ginie na wszystkich 31, PPO stoi przy ścianie do limitu akcji). Każdy z 7 dobrych runów przegrywa dokładnie te 31 i wygrywa wszystkie 169 pozostałych, czyli **169/169 wygrywalnych gier Hard**. Losowanie akcji zamiast greedy: 168/200. Ilustracja: `runs/hard-wand-in-wall.png` (seed 100000006). Wyniki Hard warto raportować także względem 169 wygrywalnych seedów.
  - Wniosek: potrzebne są oba elementy. Kształtowanie usuwa zachętę do szybkiej śmierci i daje cel pośredni, RND daje eksplorację do znalezienia minotaura z różdżką.
- Wniosek: RND działa, gdy Medium nie jest zagłuszane przez Easy i inne poziomy. Kierunki: curriculum (Easy → Medium → Hard), wyważenie poziomów w treningu wielozadaniowym, destylacja specjalistów do jednego agenta.
- Wcześniejszy wniosek (mieszanka): sam RND na tych cechach nie wystarcza; do sprawdzenia większy `rnd_int_coef`, entropia, RND z LSTM, E3B (bonus liczony w obrębie gry).

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
