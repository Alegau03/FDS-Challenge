# Todo
- z-normalizzation delle stats

- Togliere il max dalle statistiche statiche

- priority_moves combinate valutare se da togliere

- time_to_first_ko_inflicted (indice temporale del primo KO inflitto o -1 se assente) da togliere

- calcolare il win rate di ogni singolo pokemon nel dataset pesato a quante volte è presente nelle differenti battaglia

- invece che calcolare vataggio mossa per mossa, fare preprocessing sui dati estranedo i tipi dei pokemon di ogni team e calcolarsi un vataggio (con il type chart) generale del team

- si può provare a z normalizzare le statistiche statiche nel contesto del dataset non della battaglia

- nelle finestre temporali calcolare la % di hp rimasti pesati nel contesto dei pokemon che ho (se ho 3 tank ovviamente ho più %)

- possiamo provare ad indovinare i pokemon rimanenti del player 2 sapendo che il team sono bilanciati e capire se noi possiamo avere un vantaggio con i nostri pokemon rimanenti

- fare media di hp dei team e usare quella media come il 100% di vita del team (es. media generale 1000hp, se ho inflitto 300 danni ha ipoteticamente un 70%)
