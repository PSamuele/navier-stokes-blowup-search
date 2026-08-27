# AWS — guida operativa completa

Documento pratico per eseguire lo studio Run 3 su AWS senza sorprese in bolletta.
Copre: come lanciare, dove finiscono i risultati, come verificare che non ci sia
nulla di attivo, e come mettere un tetto di spesa.

Riferimento tecnico complementare: [`README_AWS.md`](README_AWS.md) (scelta istanza,
setup, benchmark).

---

## 1. In sequenza automatica o una alla volta?

**Raccomandazione: automatico in sequenza, ma solo dopo un benchmark manuale.**

Il driver (`run_convergence_R3.py`) esegue già i tre livelli uno dopo l'altro in
automatico, ed è la modalità giusta perché:

* **È ripartibile.** Ogni livello scrive `status.json`; un livello con
  `"terminated_reason": "completed"` viene saltato al riavvio. Se l'istanza cade o
  chiudi la sessione, `--resume` riprende esattamente da dove si era fermato — un
  livello a metà riparte dal suo ultimo checkpoint, non da zero.
* **Non serve sorvegliarlo.** Ogni fallimento è esplicito e scritto su disco. Il
  solutore si ferma da solo con una ragione motivata invece di continuare a produrre
  numeri privi di senso (era esattamente il problema della Run 2).
* **Va in ordine dal più economico al più costoso.** Se qualcosa non va, se ne accorge
  sul livello `coarse` dopo pochi minuti, non sul `fine` dopo trenta ore.

**Ma prima fai due passaggi manuali**, in quest'ordine:

```bash
# 1. i test devono passare sulla macchina vera (2 minuti)
cd ~/navier-stokes-cusp-study/.
python -m pytest tests -q                    # atteso: 49 passed

# 2. il livello fine non ha MAI completato un ciclo temporale (vedi sotto).
#    Provalo per pochi passi PRIMA di impegnare ore di calcolo.
python scripts/run_convergence.py --config configs/aws_production.json \
    --out_root ~/study --levels fine --meshes_only
mpirun -np 8 --bind-to core python src/solver.py \
    --mesh ~/study/meshes/apple_fine.msh --out_dir /tmp/fine_probe \
    -T 0.001 --log_interval 1 --sample_dt 1e9 --xdmf_dt 1e9 --checkpoint_dt 1e9
```

> **Perché questo passaggio è obbligatorio.** Il livello `fine` di produzione
> (752.803 celle, 4,53 M DOF di velocità) richiede ~8 GB e non è mai stato eseguito
> per intero: la macchina di sviluppo ha 8 GB di WSL e si è bloccata. Ho verificato
> separatamente che la mesh si genera e supera il controllo dimensionale, che il
> solutore si inizializza su di essa, e che SOR risolve correttamente la sua matrice
> di proiezione su 4 rank — ma il ciclo temporale su quella mesh non è mai girato.
> Questa sonda da pochi passi costa un minuto e chiude l'unica incognita rimasta.

Poi il benchmark e via:

```bash
bash deploy/run_aws.sh --ranks-sweep    # 8 o 16 rank? misura, non indovinare
bash deploy/run_aws.sh --benchmark      # proietta ore, costo e memoria
tmux new -s ns3 'bash deploy/run_aws.sh'   # studio completo
```

**Usa `tmux`.** Senza, il processo muore quando cade la connessione SSH.
Stacca con `Ctrl-b` poi `d`, riattacca con `tmux attach -t ns3`.

### Quando invece conviene uno alla volta

Solo se stai facendo diagnostica su un livello specifico:

```bash
python scripts/run_convergence.py --config configs/aws_production.json \
    --out_root ~/study --levels medium
```

---

## 1bis. Fin dove arriva davvero una run

Il solutore ha una **guardia sull'energia**: in un dominio chiuso con pareti no-slip e
senza forzante vale esattamente `dE/dt = -2 nu int|D(u)|^2 <= 0`, quindi qualunque
crescita dell'energia cinetica e' numerica per definizione. Quando l'energia risale
oltre l'1 % del suo minimo corrente, il livello si ferma e `status.json` registra
`t_at_energy_min`: l'**orizzonte affidabile**.

### Risultati reali (produzione, AWS, 8 rank)

| livello | h_pole | celle | esito | t finale | orizzonte affidabile | tempo |
| :-- | --: | --: | :-- | --: | --: | --: |
| coarse | 2,137e-3 | 47.134 | `energy_growth` | 0,2680 | **0,2631** | 5,1 min |
| medium | 1,077e-3 | 188.462 | **`completed`** | **0,5500** | **0,5500** | 1,6 h |
| fine | 5,467e-4 | 752.803 | *(in corso)* | — | — | — |

Il dato che conta: sul **medium** l'orizzonte affidabile coincide con T, cioe'
**l'energia e' decresciuta in modo monotono per tutta la simulazione**. Nessun
collasso. Il coarse invece si e' rotto a t = 0,263.

### Quanto valeva la stima a priori

Dalle sole griglie di validazione avevo estrapolato `t_collasso ~ h^-0,35`, che dava
0,243 per il coarse e 0,309 per il medium.

- Sul **coarse** la stima ha funzionato: 0,243 previsto contro 0,263 reale (errore 8 %).
- Sul **medium** ha fallito completamente: previsto un collasso a 0,309, in realta'
  **non e' mai avvenuto**.

**La lezione:** il degrado non segue una legge di potenza liscia. C'e' piuttosto una
**soglia di risoluzione**: sotto di essa la griglia perde il flusso a un certo punto,
sopra di essa il problema sparisce del tutto (almeno fino a T = 0,55). Per questo
studio la soglia sta fra `h_pole = 2,1e-3` e `1,1e-3`.

Non estrapolare l'orizzonte da due punti grossolani e poi fidarsene: e' esattamente
il tipo di ragionamento che ha prodotto i guai della Run 2. Lasciare `T = 0.55` in
configurazione e **lasciare decidere alla guardia** resta la scelta giusta, perche'
non richiede di indovinare nulla e limita il costo da sola.

### Conseguenza per l'analisi di convergenza

La finestra in cui **tutti e tre** i livelli sono validi — l'unica su cui Richardson e
GCI hanno senso — e' fissata dal piu' grossolano, quindi **t <= 0,263**.

`analyze_convergence_R3.py` tronca ogni livello al suo `t_at_energy_min`, **non**
all'ultima riga scritta: i campioni fra il minimo di energia e lo stop sono gia'
contaminati (l'energia stava salendo) e includerli significherebbe costruire
l'estrapolazione proprio sui dati non fisici che questo progetto esiste per non citare.
Lo script stampa la finestra usata per ogni livello e quale livello la limita.

Medium e fine restano confrontabili fra loro anche oltre 0,263, ma a due griglie: senza
barre d'errore GCI.

---

## 2. Dove finiscono i risultati

Tutto sotto `--out_root` (default `results/convergence_aws`):

```text
<out_root>/
├── study_config.json               parametri effettivi dello studio
├── convergence_summary.json        esito dei 3 livelli + rapporti di raffinamento misurati
├── benchmark.json                  proiezione costi (se hai lanciato --benchmark)
├── meshes/
│   ├── apple_coarse.msh  .json     mesh + statistiche REALI (h ottenuto, non richiesto)
│   ├── apple_medium.msh  .json
│   └── apple_fine.msh    .json
├── coarse/  medium/  fine/         un blocco per livello:
│   ├── blowup_data_<level>.csv     ← IL FILE PRINCIPALE. Una riga per campione:
│   │                                 t, dt, cfl, max_velocity, max_vorticity,
│   │                                 max_circulation, kinetic_energy, enstrophy,
│   │                                 div_u_l2/rel/weak, bkm_integral,
│   │                                 r/z_at_max_vorticity, bc_residual,
│   │                                 iterazioni dei 3 solve, wall_time
│   ├── status.json                 esito: completed / cfl_below_dt_min / ...
│   ├── run_meta.json               provenienza: mesh, versioni, rank, commit git
│   ├── solver.log                  stdout completo
│   ├── velocity_<level>.xdmf + .h5 ← IL FILE PESANTE (campi per ParaView)
│   └── checkpoints/                stato per rank, per il restart
└── analysis/                       generato da analyze_convergence_R3.py
    ├── convergence_report.md       ← IL RISULTATO: ordine osservato, Richardson, GCI
    ├── convergence_metrics.json    stessi numeri, leggibili da codice
    └── convergence_*.png           un grafico per grandezza
```

### Cosa riportare a casa

I file leggeri (CSV, JSON, report, PNG) sono qualche MB. Gli `.h5` sono la parte
pesante — servono solo se vuoi rivedere i campi in ParaView.

```bash
# dal TUO computer, non dall'istanza
rsync -avz --exclude='*.h5' --exclude='*.npy' \
    ubuntu@<IP>:~/navier-stokes-cusp-study/./results/convergence_aws/ \
    ./results/convergence_aws/
```

Se vuoi anche i campi, togli `--exclude='*.h5'` e metti in conto qualche GB di
transfer (~$0,09/GB in uscita).

> **Attenzione:** il `.gitignore` del repository esclude già `*.h5` e `*.npy`, quindi
> i campi pesanti non finiscono per sbaglio in un commit.

---

## 3. Verificare che non ci sia NULLA di attivo

Questa è la parte che costa soldi per distrazione. Tre categorie di trappole:

### 3a. Istanze EC2 dimenticate — **in tutte le regioni**

L'errore classico: hai avviato qualcosa in una regione, la console te ne mostra
un'altra, e non lo vedi più. Usa la vista globale:

> Console AWS → **EC2** → menu a sinistra, in cima → **EC2 Global View** →
> *Instances*

Mostra le istanze di **tutte le regioni** in una schermata sola. Deve essere vuota,
o contenere solo `terminated`.

Da riga di comando (se hai la AWS CLI):

```bash
for r in $(aws ec2 describe-regions --query 'Regions[].RegionName' --output text); do
  n=$(aws ec2 describe-instances --region $r \
        --filters "Name=instance-state-name,Values=running,pending,stopping,stopped" \
        --query 'Reservations[].Instances[].InstanceId' --output text)
  [ -n "$n" ] && echo "$r: $n"
done
echo "(nessuna riga sopra = nessuna istanza)"
```

**Importante:** un'istanza **`stopped` non costa CPU ma il suo disco EBS continua a
costare.** Fermare non basta: per non pagare più nulla devi **terminare** (Terminate),
e verificare che il volume sia sparito.

### 3b. Storage che sopravvive alle istanze

Questi restano e continuano a costare anche dopo che l'istanza non c'è più:

| Risorsa | Dove guardare | Costo tipico |
| :-- | :-- | :-- |
| **Volumi EBS** | EC2 → Volumes → filtra stato `available` (= scollegati, e li paghi comunque) | ~$0,08/GB/mese |
| **Snapshot EBS** | EC2 → Snapshots → *Owned by me* | ~$0,05/GB/mese |
| **Elastic IP** | EC2 → Elastic IPs → se **non associato**, lo paghi | ~$3,60/mese |
| **Bucket S3** | S3 → Buckets | ~$0,023/GB/mese |
| **AMI personali** | EC2 → AMIs → *Owned by me* (trattengono snapshot) | come snapshot |

```bash
# volumi scollegati in tutte le regioni
for r in $(aws ec2 describe-regions --query 'Regions[].RegionName' --output text); do
  v=$(aws ec2 describe-volumes --region $r --filters Name=status,Values=available \
        --query 'Volumes[].[VolumeId,Size]' --output text)
  [ -n "$v" ] && echo "$r: $v"
done

# Elastic IP non associati (si pagano proprio perché inutilizzati)
for r in $(aws ec2 describe-regions --query 'Regions[].RegionName' --output text); do
  e=$(aws ec2 describe-addresses --region $r \
        --query 'Addresses[?AssociationId==`null`].PublicIp' --output text)
  [ -n "$e" ] && echo "$r: $e"
done
```

### 3c. La verifica che conta davvero: la fattura

Tutto il resto è indiziario. **Questa è la prova.**

> Console AWS → cerca **Billing and Cost Management** → **Bills**

Seleziona il mese corrente e apri **"Charges by service"**. Se qualcosa sta costando,
compare qui, qualunque sia la regione o il servizio. Espandi ogni voce per vedere in
quale regione sta accadendo.

Poi, per l'andamento nel tempo:

> **Cost Explorer** → *Daily* → raggruppa per *Service*

Un grafico giornaliero a zero (o solo credito) è la conferma definitiva che non stai
pagando niente. Nota: Cost Explorer si aggiorna con **fino a 24 ore di ritardo**, quindi
subito dopo aver terminato un'istanza vedrai ancora il costo di ieri.

### Checklist di chiusura, dopo lo studio

```
[ ] risultati scaricati con rsync e verificati in locale
[ ] EC2 Global View: nessuna istanza (o solo "terminated")
[ ] EC2 → Volumes: nessun volume "available"
[ ] EC2 → Snapshots / AMIs: solo quello che vuoi tenere
[ ] EC2 → Elastic IPs: nessuno non associato
[ ] Billing → Bills: nessun servizio inatteso
[ ] fra 24 h: Cost Explorer daily torna a zero
```

---

## 4. Non spendere oltre il bonus di €200

### Il punto fondamentale, da sapere prima di tutto

> **AWS non ha un tetto di spesa che blocca automaticamente i servizi.**
> I "Budget" sono **avvisi**, non limiti. Se superi la soglia ricevi una mail, ma
> l'istanza continua a girare e a costare.

L'unico meccanismo che *ferma* davvero qualcosa sono le **Budget Actions**, che
applicano automaticamente una policy IAM restrittiva al superamento della soglia.
Va configurato esplicitamente — vedi 4c.

### 4a. Controlla prima i crediti

> Billing and Cost Management → **Credits**

Verifica tre cose, perché i crediti promozionali hanno vincoli:

1. **Data di scadenza** — spesso 12 mesi, e i crediti scaduti non tornano.
2. **Servizi ammessi** — alcuni crediti valgono solo su certi servizi. EC2 ed EBS di
   norma sono coperti; controlla.
3. **Saldo residuo**.

I crediti si applicano automaticamente prima del metodo di pagamento. **Quando
finiscono, AWS addebita la carta senza chiedere conferma** — da qui l'importanza di
quanto segue.

### 4b. Budget con avvisi (fallo subito, 5 minuti)

> Billing → **Budgets** → *Create budget* → **Customize (advanced)** → *Cost budget*

Configurazione consigliata:

| campo | valore |
| :-- | :-- |
| Periodo | Monthly, ricorrente |
| Importo | **150 USD** (margine sotto i ~215 USD di €200) |
| Scope | tutti i servizi |
| **Voce chiave** | in *Budget scope* seleziona **"Include credits"** o equivalente: vuoi essere avvisato sul consumo **lordo**, non su quanto resta da pagare dopo i crediti |

Poi aggiungi **più soglie**, non una sola:

| soglia | tipo | cosa significa |
| :-- | :-- | :-- |
| 50 % | Actual | "sto consumando il bonus" |
| 80 % | Actual | "attenzione" |
| 100 % | **Forecasted** | "alla proiezione attuale sforo" — **è l'avviso più utile**, arriva prima |
| 100 % | Actual | "bonus esaurito, da qui paghi tu" |

Metti la tua mail su tutte.

Crea anche il **zero-spend budget**: AWS offre un template preconfigurato che avvisa
appena la spesa supera $0,01. È il modo migliore per accorgersi di risorse dimenticate
quando pensi di aver chiuso tutto.

### 4c. Il tetto vero: Budget Action

Questa è la cosa che effettivamente **impedisce** di spendere oltre.

> Nel budget appena creato → **Add action**

| campo | valore |
| :-- | :-- |
| Soglia | 90 % di actual (quindi ~$135) |
| Action type | **IAM policy** |
| Policy | una policy che nega l'avvio di risorse |
| Approvazione | scegli **automatica** se vuoi il blocco vero; "manuale" richiede che tu confermi via mail |

Policy IAM da allegare (crea in IAM → Policies → JSON):

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Deny",
    "Action": [
      "ec2:RunInstances",
      "ec2:StartInstances",
      "ec2:CreateVolume"
    ],
    "Resource": "*"
  }]
}
```

**Limite importante, da capire bene:** questa policy impedisce di **avviare cose
nuove**, ma **non spegne un'istanza già in esecuzione**. Per fermare anche quella serve
un'azione di tipo *EC2 Instance* nella Budget Action (che ferma le istanze indicate),
oppure una Lambda. Per uno studio che dura ~15 ore su una sola istanza, la combinazione
avvisi + policy di Deny + la checklist di chiusura è ampiamente sufficiente.

### 4d. Disciplina operativa (vale più di qualsiasi configurazione)

1. **Un'istanza sola, una regione sola.** Segnati quale. Non aprire risorse altrove.
2. **Fai `--benchmark` prima dello studio.** Se proietta più di ~100 ore, fermati e
   riduci `T` o la risoluzione invece di pagare.
3. **Termina appena hai scaricato i risultati.** Un `c6i.4xlarge` dimenticato costa
   ~$16 al giorno: due settimane di distrazione e il bonus è finito.
4. **Considera l'auto-shutdown.** Se temi di dimenticartene, avvia l'istanza con questo
   *user data*, che la spegne dopo 24 ore qualunque cosa accada:

   ```bash
   #!/bin/bash
   shutdown -h +1440
   ```

   Con *Shutdown behavior* impostato su **Terminate** (nelle impostazioni avanzate
   dell'istanza), a quel punto sparisce anche il volume EBS.

### Costo atteso di questo studio

| voce | stima |
| :-- | --: |
| calcolo, `c6i.4xlarge` on-demand, ~15 h | ≈ $10 |
| EBS 60 GB gp3 per ~2 giorni | ≈ $0,40 |
| transfer in uscita (poche centinaia di MB) | ≈ $0,05 |
| **totale** | **≈ $11** |

Anche a 5× il tempo previsto resti sotto $60. Il bonus non è il vincolo: il rischio
reale è la risorsa dimenticata, non lo studio.

---

## 5. Riepilogo comandi

```bash
# --- setup (una volta) ---
ssh ubuntu@<IP>
git clone https://github.com/PSamuele/Navier_Stokes_Cusp_Study.git ~/navier-stokes-cusp-study
cd ~/navier-stokes-cusp-study
git checkout run-03-corrected-solver-and-convergence-study   # finché non è in main
bash ./deploy/aws_setup.sh
conda activate fenicsx-env

# --- verifica (obbligatoria) ---
cd .
python -m pytest tests -q                       # 49 passed
bash ../deploy/run_aws.sh --ranks-sweep         # 8 o 16 rank
bash ../deploy/run_aws.sh --benchmark           # ore e costo previsti

# --- studio ---
tmux new -s ns3 'bash ../deploy/run_aws.sh'
#   stacca: Ctrl-b poi d      riattacca: tmux attach -t ns3
#   dopo un'interruzione:     bash ../deploy/run_aws.sh --resume

# --- risultati ---
cat ~/navier-stokes-cusp-study/./results/convergence_aws/analysis/convergence_report.md

# --- dal tuo computer ---
rsync -avz --exclude='*.h5' --exclude='*.npy' \
    ubuntu@<IP>:~/navier-stokes-cusp-study/./results/convergence_aws/ \
    ./results/convergence_aws/

# --- poi TERMINA l'istanza e fai la checklist della sezione 3 ---
```
