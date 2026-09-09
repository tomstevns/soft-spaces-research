# Soft Spaces fase 2 — læsevejledning til v25.33–v25.41

**Formål:** En rolig indføring i, hvad iterationerne faktisk viser, hvorfor den
numeriske stabilitet er så stor, og hvor den matematiske grænse nu ligger.

## 1. Udgangspunktet

De oprindelige 8Q–12Q-kørsler viser et reproducerbart numerisk mønster. Den
fastlåste regel

\[
k_{q+1}=2k_q+1
\]

ramte de på forhånd valgte koordinater, og i 12Q blev begge frosne kandidater
bekræftet med positiv prominens i 6 af 8 uafhængige batches, altså præcis ved
den forud fastsatte grænse på (0,75).

Det er **empirisk evidens**. Det er endnu ikke et generelt matematisk bevis for,
at hotspotstyrken altid bevares.

## 2. Det vigtigste ved koordinaten (k)

I 8Q–12Q-programmerne er (k) placeringen af et nabopar i den sorterede liste
af egenværdier og egenvektorer. Det er ikke nummeret på en computational
basis-tilstand.

Derfor er

\[
k\longmapsto 2k+1
\]

en eksakt aritmetisk indeksregel, men den giver ikke automatisk en fysisk
Hilbert-rumsoperator

\[
W|k\rangle=|2k+1\rangle.
\]

En sådan operator kræver, at (|k\rangle) betegner faktiske tilstande i et
fastlagt basis. I målingerne betegner (k) derimod en spektral rang, som først
opstår efter diagonaliseringen af hver ny Hamiltonian.

## 3. Iterationerne i almindeligt sprog

| Iteration | Spørgsmål | Resultat | Status |
|---|---|---|---|
| v25.33 | Bevares hele score og prominens ved en perfekt kopieret, afkoblet ancilla? | Ja, til ca. (1,3\times10^{-14}) i implementeringskontrollen. | Eksakt betinget sætning; numerisk verificeret implementering |
| v25.34 | Hvad sker der, når ancillaen kobles kontrolleret til systemet? | Stor stabilitet på det frosne gitter; en positiv seed mistede dog den tilstrækkelige betingelse ved meget stærk kobling. | Numerisk stresstest |
| v25.35 | Kan almindelige perturbationsgrænser bevise stabiliteten? | Ja, men den fælles sikre radius blev ekstremt lille. | Rigorøs, meget konservativ grænse |
| v25.36 | Hvorfor er de numeriske værdier langt mere stabile end den første grænse? | Førsteordensændringen forsvinder; responsen starter kvadratisk. | Eksakt symmetri plus numerisk mekanismeaudit |
| v25.37 | Kan symmetrien give en skarpere andenordensgrænse? | Den forklarer stabiliteten langt bedre, men et endeligt kontinuumsbevis manglede. | Delvist analytisk, delvist numerisk |
| v25.38 | Kan alle nødvendige afledte og marginer afgrænses på et helt interval? | Ja for frosne instanser, men de første beregninger brugte ikke fuld intervalaritmetik. | Kandidatcertifikat, endnu ikke endeligt |
| v25.39 | Holder certifikatet med udadrettet Arb-intervalaritmetik? | Ja: tre positive seeds forbliver positive for (|\rho|\le8,9\times10^{-11}); to negative seeds forbliver negative. | Computerassisteret intervalbevis for de frosne instanser |
| v25.40 | Er den faktiske 12Q-generator den ancillakopi, som beviset kræver? | Nej i den direkte, naturlige samme-seed-test: 0/40 kopier, 0/200 kopierede Pauli-led og 0/40 ancillainvariante Hamiltonianer. | Eksakt kodeaudit og endelig finit test |

## 4. Hvorfor den numeriske stabilitet er så stor

Den kontrollerede ancillamodel har en paritetssymmetri. Fortegnet af koblingen
kan vendes ved en unitær transformation på ancillaen:

\[
H(+\epsilon)\sim H(-\epsilon).
\]

En spektral score, som er uændret under denne transformation, bliver derfor en
lige funktion:

\[
\Pi(+\epsilon)=\Pi(-\epsilon).
\]

Taylorudviklingen kan så ikke indeholde et lineært led:

\[
\Pi(\epsilon)
=\Pi(0)+c_2\epsilon^2+c_4\epsilon^4+\cdots.
\]

Det er hovedforklaringen. Når koblingen gøres ti gange større, vokser den
første mulige ændring omtrent hundrede gange — ikke ti gange — så længe man
forbliver i det lokale, glatte regime.

Tre yderligere forhold hjælper:

1. De to ancillagrene er spektralt adskilt, så koblingen skal overvinde en
   reel energiseparation, før tilstandene blandes kraftigt.
2. Scorefunktionen er normaliseret. Fælles skalaforandringer i tæller og nævner
   ophæver derfor delvis hinanden.
3. Prominensen er en forskel mellem kandidat og lokale kontroller. Hvis begge
   flytter sig på lignende måde, kan selve forskellen være mere stabil end de
   enkelte rå værdier.

Det forklarer den observerede stabilitet i **den kontrollerede ancillamodel**.
Det beviser ikke, at præcis samme mekanisme skaber 8Q–12Q-mønstret.

## 5. Hvad v25.39 virkelig beviser

For de tre frosne instanser, som var positive ved nul kobling, er følgende
computerassisteret bevist med udadrettet intervalaritmetik:

\[
|\rho|\le 8,9\times10^{-11}
\quad\Longrightarrow\quad
\Pi(\rho)>0.
\]

Det betyder ikke, at (8,9\times10^{-11}) er den fysiske brudgrænse. Tallet er
en garanteret fælles radius fra konservative uligheder. De almindelige
flydende-punktsberegninger viser stabilitet meget længere ude, men dette større
område er ikke bevist som et helt kontinuum.

De to oprindeligt negative seeds blev ikke fjernet eller omklassificeret. De
blev intervalvalideret som negative og indgår fortsat i dokumentationen.

## 6. Hvad v25.40 ændrer

v25.40 ændrer ingen tidligere observationer. Den afgrænser forklaringen.

12Q-programmet genererer en ny fem-leddet random-Pauli-Hamiltonian direkte i
12Q. Selv når man giver 11Q- og 12Q-generatoren samme seed og dermed den mest
fordelagtige simple kobling, bliver Pauli-etiketterne dimensionsafhængigt
gensamplet.

Resultatet for de 40 frosne 12Q-seeds var:

- 0 af 40 var en eksakt (I_a\otimes H_{11})-kopi;
- 0 af 200 Pauli-led var eksakte kopier af det tilsvarende 11Q-led;
- 0 af 40 bevarede begge simple computational-ancillasektorer.

Dermed er den direkte bro fra v25.33–v25.39 til de oprindelige uafhængige
8Q–12Q-Hamiltonianer **afkræftet for denne naturlige konstruktion**.

Det er ikke en afkræftelse af hotspotmønstret. Det betyder, at vi nu har to
ærligt adskilte resultater:

1. en statistisk, reproducerbar 8Q–12Q-rekurrence;
2. et rigorøst stabilitetsbevis for en kontrolleret ancillamodel.

## 7. Forbindelsen til Breuer og Petruccione

På de fotograferede sider i kapitel 9 defineres projektionssuperoperatorerne

\[
\mathcal P\rho=\operatorname{tr}_B(\rho)\otimes\rho_B,
\qquad
\mathcal Q\rho=\rho-\mathcal P\rho,
\]

og Nakajima–Zwanzig-ligningen udledes med en hukommelseskerne. Det centrale
princip er præcis relevant: man deler dynamikken i en relevant (mathcal P)-del
og en komplementær (mathcal Q)-del og undersøger tilbagevirkningen fra
(mathcal Q) på (mathcal P).

Men bogens (mathcal P) og (mathcal Q) virker på densitetsmatricer i
Liouville-rummet. Fase-2-leddet med en resolvent og to-dimensionale spektrale
par ligger i en nært beslægtet Feshbach/Schur-komplement-tradition. De må ikke
identificeres ordret uden en særskilt afledning.

## 8. Den næste ærlige forskningsretning

Det næste spørgsmål er ikke at presse ancillabeviset ind over de gamle data.
Det er at undersøge, om den uafhængige random-Pauli-ensemble selv har en
**fordelingsmæssig eller spektral selvlighed**, som kan forklare, hvorfor
bestemte relative spektrale placeringer gentager sig.

En sådan undersøgelse skal være præregistreret og skal mindst skelne mellem:

- absolut indeks og normaliseret spektral position;
- konsekvenser af sortering og naboparsdefinition;
- random-Pauli-ensemblets dimensionsskalering;
- den valgte REAL–NULL-normalisering;
- hotspotværdi og lokal baggrund;
- et ægte signal og en artefakt fra selve scoringsproceduren.

Først hvis denne forbindelse kan bevises eller klart afkræftes, er der ikke
mere at hente i den nuværende matematiske retning.

## 9. Kort konklusion

Den store numeriske stabilitet er reel i den kontrollerede model og har en
klar hovedmekanisme: ancillapariteten fjerner førsteordensresponsen. v25.39 gør
en lille, men ubestridelig del af stabilitetsområdet rigorøst. v25.40 viser
samtidig, at de oprindelige 8Q–12Q-kørsler ikke blev genereret af den direkte
ancillakopi, som dette bevis forudsætter.

Det svækker ikke de registrerede data. Det forhindrer en for stærk forklaring
og gør næste spørgsmål langt skarpere.

## 10. v25.41 — forklaringen på selve høj-Q-progressionen

v25.41 fandt den forbindelse, som v25.40 viste ikke kunne være en direkte
ancillakopiering.

Reglen

\[
k_{q+1}=2k_q+1
\]

er identisk med bevarelse af den normaliserede spektrale position:

\[
\frac{k_{q+1}+1}{2^{q+1}}=\frac{k_q+1}{2^q}.
\]

Kæderne følger derfor præcist (1/4) og (3/4) af kildespektret. De to
targetgrene placerer de fire undersøgte nabopar ved de ulige ottendedele.

Samtidig består hver Hamiltonian kun af fem Pauli-led. Deres operatoralgebra
forbliver lille, mens resten af Hilbertrummet bliver et stadig større
identitets- eller tilskuerrum. Det tvinger store multipliciteter ind i spektret.
Ved 12Q var den tvungne egenværdimultiplicitet 512 for 35 af 40 seeds og 256
for de øvrige fem.

De frosne kandidater ligger præcis på grænserne mellem disse gentagne
spektralblokke. Alle de frosne 12Q-kontrolpar ligger derimod inde i blokkene og
har derfor eksakt nul nabogab.

Den faste kontrolradius forklarer også høj-Q-forstærkningen:

| Target | Gentagelseslængde | Kontroller tvunget ind i flade blokke |
|---:|---:|---:|
| 7Q | 4 | 6/10 |
| 8Q | 8 | 8/10 |
| 9Q–12Q | mindst 16 | 10/10 |

De to 12Q-kandidater var desuden ikke uafhængige. De var positive i nøjagtigt
de samme seks batches og negative i de samme to, brugte identiske kvalificerede
seedmængder og havde batchkorrelation (0,9894). For én kandidat er
sign-testens hale for mindst 6 af 8 positive batches (0,1445).

Den korrekte slutning er derfor:

> Den registrerede høj-Q-progression er forklaret som en konsekvens af den
> fem-leddede Pauli-algebras multipliciteter, spektral sortering og den valgte
> kandidat/kontrol-geometri. Den dokumenterer ikke transport af et fysisk
> hotspot mellem uafhængige Hilbertrum.

Dataene er stadig korrekte og værdifulde, men deres fortolkning er nu mere
begrænset. En ny fysisk undersøgelse skal bruge et ensemble, hvis antal eller
lokalitet af led skalerer med (q), og kontroller på samme type spektrale
blokgrænser som kandidaterne.

## 11. v25.42 — den afgørende basisinvarians-test

En degenereret egenværdi har ikke entydige egenvektorer. Hamiltonianen
bestemmer hele egenrummet, men enhver unitær rotation inde i det giver en lige
så korrekt egenbasis og efterlader Hamiltonianen uændret.

v25.42 fastholdt derfor Hamiltonian, spektrum, perturbation, kandidater,
kontroller og score og roterede kun egenbasen inde i de degenererede blokke.
Testen brugte tolv frosne 8Q-Hamiltonianer og otte frosne rotationer pr. basis.

Rekonstruktionsfejlen var højst

\[
1,44\times10^{-14}
\]

for REAL og (9,21\times10^{-15}) for NULL. Hamiltonianerne var således de
samme inden for almindelig flydende-punktspræcision.

Alligevel skiftede hotspotprominensen fortegn:

| Rotation | Fortegnsområde krydsede nul | Baselinefortegn ændret |
|---|---:|---:|
| REAL-basis | 2/10 kvalificerede tilfælde | 2/10 |
| NULL-basis | 4/10 kvalificerede tilfælde | 4/10 |

Et enkelt fortegnsskift er nok som modeksempel. Scoren er derfor ikke invariant
under fysisk ækvivalente basisvalg i de store degenererede egenrum.

Den endelige konklusion for den oprindelige fem-leddede høj-Q-model er:

> (2k+1)-progressionen skyldes spektral blokgeometri, og den resterende
> hotspotklassifikation afhænger af diagonaliseringsprogrammets tilladte valg
> af egenbasis. Den er derfor ikke et entydigt fysisk observable for
> Hamiltonianen.

Flere seeds med samme protokol kan ikke løse dette. En fortsættelse kræver en
ny model: enten komplette degenererede egenrumsprojektorer eller en fysisk
Hamiltonian med tilstrækkeligt mange skalerende lokale led til at ophæve den
kunstige tilskuerdegeneracitet.
