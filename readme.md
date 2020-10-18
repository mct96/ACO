Introdução {#sec:intro}
==========

*Ant Colony Optimization* (ACO) pertence a uma família de algoritmos
baseados em *swarm intelligence*. Estes algoritmos são baseados em
comportamentos observados em diversas espécies de animais, como cupins,
formigas, aves, peixes, entre outros, onde, embora cada indivíduo não
apresente inteligência significativa, quando reunidos em colônias há o
surgimento de inteligência nas ações coletivas [@Castro2006].

O ACO, por sua vez, tem como metáfora o comportamento da colônia de
formigas. Mais especificamente, este algoritmo abstrai o método que as
formigas utilizam no processo de busca por alimento, onde elas utilizam
de um processo bioquímico denominado feromônio que, somado fatores
implícitos, influenciam a decisão das demais, propriedade conhecida como
*estigmergia*.

O ACO é aplicado para resolver problemas NP-*complete* e combinatoriais
conforme proposto por diversos autores: @Luo2014, @Asif2009. Isso ocorre
pelo fato de que os agentes (neste caso as formigas) exploram o espaço
de soluções influenciadas pela qualidade dos resultados passados,
indicado pelo quantidade de feromônio no caminho, ou seja, há uma
exploração guiada na direção de bons resultados passados, o que causa a
convergência da população no decorrer das gerações.

Um aspecto importante do ACO e de todos os algoritmos evolucionários
está na modelagem do problema. Em [@Dorigo1991] os autores aplicam o ACO
no problema do caxeiro viajante (TSP, do inglês *Travelling Salesman
Problem*), que consiste em encontrar o menor caminho que percorre todas
as cidades apenas uma vez. Nesse artigo, os autores modelaram o problema
em forma de grafos, onde cada cidade é um nó e cada caminho entre
cidades é um grafo não direcionado. Neste trabalho uma abordagem
semelhante é adotada, porém diferente da abordagem tradicional do TSP,
que busca o menor caminho que passe por todas as cidades um única vez,
neste trabalho é feito o inverso: o maior caminho que percorre todas as
cidades apenas uma vez. O fato de que uma cidade pode ser visitada
apenas uma vez é uma restrição é específica do problema e não é
especificado pelo algoritmo em sua forma canônica. Portanto, é
necessário a elaboração de mecânismos que tornem o algoritmo genérico
capaz de atuar em situações como essa.

Neste trabalho é proporto dois mecânismos para resolver o problema de
encontrar a rota com o maior custo sem que o mesmo nó seja visitado mais
de uma vez. Para isso, o método de solução e a modelagem desenvolvida
para o TSP em [@Dorigo1991] é adotada, porém com ajustes para a
maximização ao invés de minização do custo do caminho. Neste trabalho é
também introduzido dois algoritmos para o problema de geração de
soluções candidatas, o *Backoff* e o *Stochastic Backoff*, onde o
primeiro é um caso específico do último, que ocorre quando o parâmetro
$\xi$ é igual a $0$.

O restante deste trabalho é estruturado da seguinte maneira: na é
apresentado como o problema é modelado e como o algoritmo proposto por
@Dorigo1991 e ajustado para obter a maximização ao invés da minimização
do custo. Nessa seção também é descrito os algoritmos para lidar com
soluções inválidas e quais são os experimentos realizados para verificar
o desempenho deles. Na é apresentado os resultados do experimentos
realizados, como a média e o desvío padrão do custo encontrado pela
população no decorrer das gerações. Por fim, na a performance do
algoritmo é analisado e trabalhos futuros são propostos.

Método e Experimentos {#sec:met_and_exp}
=====================

Nesta seção é descrito os datalhes de implemetação e modelagem do
problema abordado neste trabalho. A princípio, o método desenvolvido por
@Dorigo1991 é descrito, após isso, é apresentado os ajustes para adaptar
as funções utilizadas em um problema de minimização para um problema de
maximização.

Modelagem do problema
---------------------

Para gerar uma formiga, um nó é escolhido aleatóriamente com
probabilidade uniforme. Após isso, a formiga passa a construir uma
solução decidindo qual nós visitar a partir da vizinhaça do nó em que
ela se encontra. Essa decisão é feita considerando dois fatores:

a quantidade de feromônio contido no grafo que conecta o nó atual com o
avaliado e

a distância (euclidiana, neste trabalho) entre os nós

. A equação que sumariza a probabilidade de que um determinado nó na
vizinhaça será visitado é:

$$\label{eq:prob}
  p_{ij}^{k}(t)=
  \begin{cases}
    \frac{[\tau_{ij}(t)]^\alpha\,\cdot\,[\eta_{ij}]^\beta}{\sum_{l \in J_i^k}[\tau_{il}(t)]^\alpha\,\cdot\,[\eta_{il}(t)]^\beta} & \quad \textrm{se j está na vizinhaça de i} \\
    0 & \quad \textrm{caso contrário}
  \end{cases}$$ onde,

  ------------------------------- -------------------------------------------------------------------
  $\tau_{ij}$                     quantidade de feromônio entre o nó i e j
  $\eta_{ij}$                     é a visibilidade do nó i a partir do j
  $J_i^k$                         a *tabu list* de cidades a serem visitadas pela formiga $k$
  $\textrm{$\alpha$ e $\beta$}$   são parâmetros que regulam o impacto da distância ou do feromônio
  ------------------------------- -------------------------------------------------------------------

A é mantida neste trabalho sem alterações em relação ao algoritmo
original. Todos os parâmetros contidos nessa equação são possíveis de
serem variados e sob a mesma interpretação e impacto identificados pelo
trabalho de [@Dorigo1991] e discutido em [@Castro2006]. A *tabu list*
descrita na equação se refere a lista de cidades que ainda não foram
visitadas pela formiga.

Outra equação importante e que é responsável pela convergência do
algoritmo é a atualização do feromônio contido em um determinado grafo
que conecta os dois nós, i e j, que é representado originalmente pela
seguinte equação:

$$\label{eq:feromonio_inc}
    \Delta\tau_{ij}^k(t) =
    \begin{cases}
      Q/L^k(t) & \quad \textrm{se $(i, j) \in T^k(t)$}\\
      0 & \quad \textrm{caso contrário}
    \end{cases}$$ onde,

  ---------- -------------------------------------------------------------
  $L^k(t)$   é o custo do caminho $T^k(t)$ desenvolvido pela formiga $k$
  $Q$        é um ganho a ser definido pelo usuário
  ---------- -------------------------------------------------------------

Na o custo do caminho é o denominador da equação, o que faz com que
quanto maior o custo, menor o incremento de feromônio nesse caminho.
Esse é o comportamento oposto ao desejado neste trabalho. Portanto, a é
adaptada e o custo, denotado por $L^k(t)$, passa a ser multiplicado pela
constante $Q$. A apresenta a modificação, porém todos os componentes
ainda possuem o mesmo significado. A convergência é alcançada por causa
do decaimento do feromônio e pelo incremento de feromônio ao longo das
gerações. Para rotas que pouco incremento, a probabilidade de ser
selecionada pela formiga é pouco provável.

$$\label{eq:feromonio_inc_adapted}
    \Delta\tau_{ij}^k(t) =
    \begin{cases}
      Q \, \cdot \, L^k(t) & \quad \textrm{se $(i, j) \in T^k(t)$}\\
      0 & \quad \textrm{caso contrário}
    \end{cases}$$

Outro detalhe importante para a modificação feita na equação apresentada
acima é que após gerar os valores de atualização de todos os grafos,
eles são normalizados para evitar que os nós possuam uma proporção
desequilibrada entre si.

A atualização em si é trivial é pode ser apresentada na . Essa equação
possui um parâmetro, $\rho$, que representa a taxa de decaimento de
feromônio. Valores próximos de 1 significam que pouca informação das
interações anteriores são guardadas, já os valores próximos de 0 indicam
que o progresso anterior deve ser mantido. O ajuste desse parâmetro
afeta dois aspectos conflitantes dos algoritmos evolucionários:
*exploration* e *explotation*.

$$\label{eq:feromonio_up}
    \tau_{ij}(t) \gets (1-\rho)\tau_{ij}(t) + \Delta\tau_{ij}(t)$$ onde,

  ---------------------- -------------------------------------------------------------
  $\rho$                 [taxa de dacaimento do feromônio $\rho \in [0, 1)$]{.roman}
  $\Delta\tau_{ij}(t)$   $\Delta\tau_{ij}(t) = \sum_k \Delta \tau^k_{ij}(t)$
  ---------------------- -------------------------------------------------------------

Outro conceito introduzido por @Dorigo1991 foi o conceito de formiga
elitista, que é representada pela melhor solução encontrada até o
momento. Após atualizar o feromônio de todos os grafos, aqueles que
contituem a melhor solução até o momento recebem um reforço de
feromônio. Esse reforço tende guiar a colônia em busca da melhor
solução. A representação matemática desse conceito está na . Note que o
ajuste do parâmetro $b$ intensifica ou atenua o efeito do elitismo.

$$\label{eq:eletist}
    \Delta\tau_{ij}^k(t) = b\,\cdot\,Q\,\cdot\,L_{best}$$ onde,

  ------------ -----------------------------------------------------------
  $b$          é um parâmetro definido pelo usuários
  $Q$          é o parâmetro de ganho da atualização, descrito acima
  $L_{best}$   é o percurso percorrido pela melhor solução até o momento
  ------------ -----------------------------------------------------------

Na equação acima, em sua versão original, o componente $L_{best}$ divide
as constantes, porém aqui ele foi adaptado para a maximização do custo.

*Backoff*
---------

O mecânismo de *backoff* se baseia no conceito de transmissão de dados.
Quando um dispositivo tenta transimitir dados e ocorre algum erro no
processo ao invés de repetir a transmissão imediatamente, os
dispositivos que utilizam o método de acesso CSMA/CA aguardam por
determinado período de tempo e tentam novamente. Caso ocorra outro erro,
o equipamento aguarda outro momento, porém dessa vez maior que o
intervalo anterior. Esse macânismo visa evitar o congestionamento da
rede.

Este trabalho utiliza esse conceito para a elaboração de soluções, isto
é, a trajetória selecionada pelos agentes (formigas na metáfora do ACO).
A necessidade desse método surge do seguinte problema: a formiga avança
pelos nós até o momento em que todas os nós vizinhos já foram visitados
e ainda há nós na *tabu list* a serem visitados. Nessa situação, uma
possível abordagem é reiniciar a formiga, porém, para grafos muito
grandes, o custo computacional descartado pode ser muito elevado, uma
vez que, é mais provável que a formiga fique sem movimentos válidos
apenas quando muitas cidades já tiverem sido visitadas. E é nesse
contexto que o *backoff* é introduzido.

A aplicação do algoritmo é da seguinte forma: Quando não houver
movimentos válidos, porém a *tabu list* não estiver vazia, recue $b$
movimentos e busque a solução novamente. Caso as situação descrita
anteriormente se repita, recue $b+1$ movimentos. Eventualmente, caso o
*backoff* ocorra demasiadamente, a formiga voltará a posição inicial e,
neste caso, um novo ponto de partida é gerado aleatóriamente.

Uma suposição que ocorre ao adotar o *backoff* é que há uma solução que
passe por todos os nós uma única vez, porém isso não pode ser garantido,
uma vez que, um nós pode ter apenas um único vizinho e, neste caso, esse
grafo só faz parte da solução se ele for o ponto de partida ou de
término da solução. Quando há dois nós no grafo com apenas um vizinho,
há apenas um solução (desconsiderando-se soluções que variam apenas pela
direção em que são percorridas). Para o caso de três ou mais nós com
apenas um vizinho, é impossível encontrar uma solução que percorre todos
os nós. Nesse último caso, o *backoff* no seu modelo original, não
retornaria nenhuma solução, pois ele sempre reiniciaria a formiga em
outra posição a procura de uma rota que visitasse todos os nós.

*Stochastic Backoff*
--------------------

Nesse contexto que surge a versão geral do algoritmo baseado em
*backoff*, denominado *stochastic backoff*. O termo *stochastic* se deve
ao fato que o *backoff* ocorre probabilisticamente, baseado em uma
função $p(b)$, definida da seguinte maneira:

$$p(b) = e^{\xi(b - 1)}$$ onde,

  ------- --------------------------------------
  $b$     quantidade de *backoff* já ocorridos
  $\xi$   coefiente de decaimento
  ------- --------------------------------------

Quando uma formiga não possui rotas e ainda há nós não visitados, antes
de realizar o *backoff*, ela toma uma decisão de tentar recuar e tentar
novamente (no contexto da metáfora). Caso ela recue, o progresso dela é
salvo em um arquivo que contém o caminho que ela percorreu e o custo
associado a cada tentativa. Em algum momento, conforme a quantidade de
*backoff* aumenta, a probabilidade de ela tentar novamente é reduzido
até a desistência. Quando isso ocorre o melhor caminho encontrado é
retornado como um possível solução. No contexto de problemas de
maximização, o melhor caminho é o que possui o maior custo.

Parâmetros
----------

Essas equações constituem o mecânismo responsável pela estrutura do
algoritimo. Na estão contidos todos os parâmetros customizáveis neste
trabalho. O valor ideal de cada um deles é específico do problema em que
ele é aplicado.

  ---------- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  $\alpha$   Este parâmetro é responsável por controlar o impacto que a quantidade de feromônio tem na escolha de rotas. Isso faz com que a formiga tenha maior probabilidade de selecionar vértices que geraram bons resultados em iterações passadas.
             
  $\beta$    Semelhante ao $\alpha$, porém controla o impacto da visibilidade do nó ($\eta_{ij}$). Isso faz com que a formiga tenha maior probabilidade de mover para um nó que possui maior custo.
             
  $\rho$     Regula a taxa de decaimento do feromônio. Tem impacto no *explotation*/*exploration*.
             
  N          O tamanho da população. Quanto maior a população, há maior possibilidade de exploração do espaço de busca e de encontrar boas soluções.
             
  Q          Regula a amplitude da atualização de feromônio.
             
  $\tau_0$   O valor inicial do feromônio contido nos grafos. Esse valor deve ser diferente de 0.
             
  b          É o fator que regula o impacto das formigas elitistas.
             
  $\xi$      O coeficiente de decaimento. Deve estar no intervalo $[0, +\infty)$. Ajusta a probabilidade do *blackoff* ocorrer.
  ---------- --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

  : Parâmetros do algoritmo[]{label="tbl:parameters"}

Para executar o algoritmo o comando representado a seguir de ser usado.
As opções estão especificadas na , onde o parâmetro especificado na
implementação é associado ao parâmetro que ele representa no algoritmo.

`python3 ACO.py <dataset> <output> <#formigas> <#iterações> [options]`

Por exemplo, para utilizar o arquivo entrada1.txt, salvar os resultados
em ACO\_res.csv com 40 formigas, 50 iterações, feromônio inical igual a
$0.6$, taxa de decaimento de feromônio igual a $0.5$, $\alpha$ igual a
1, $\beta$ igual a 5, ganho na amplitude da atualização igual a $5$,
reforço na melhor trajetória encontrada igual a $5$ e coeficiente de
decaimento igual a $0.3$:

`python3 ACO.py ../dataset/entrada1.txt  /ACO_res.csv 40 50 –initial-pheromone 0.6 –decay-rate 0.5 –alpha 1 –beta 5 –reinforcement-gain 5 -e –eletism-gain 5 –xi 0.3`

[\[tbl:code\_par\]]{#tbl:code_par label="tbl:code_par"}

  ---------- -------------------------------------------------------
  $\alpha$   `–alpha` \[valor\]
  $\beta$    `–beta` \[valor\]
  $\rho$     `–decay-rate` \[valor\]
  Q          `–reinforcement-gain` \[valor\]
  $\tau_0$   `–initial-pheromone` \[valor\]
  b          `–eletism-gain` \[valor\] (depende da flag --eletism)
  $\xi$      `–xi` \[valor\]
  ---------- -------------------------------------------------------

  : Parâmetros de implementação

Resultados {#sec:result}
==========

Os resultados apresentados nessa seção são o resultado a iteração do
algoritmo diversas vezes e as médias dos parâmetros são retornados. Como
há 8 parâmetros distintos e 3 datasets, apenas quatro parâmetros são
analisados: o tamanho da população, a taxa de decaimento do feromônio, o
coeficiente de decaimento do *backoff* e a presença de elitismo. As
figuras contidas nessa seção apresentam o valor da média (linha
contínua) e do desvio padrão (linha tracejada) ao longo das iterações.

Dataset 1
---------

O primeiro teste verifica o impacto do tamanho da população nas soluções
encontradas. Nesse teste é também comparado o uso de *stochastic
backoff* juntamente com o *backoff* com os mesmos parâmetros, com
excessão do $\xi$, que não está presente no último. Os valores
experimentados são: $10$, $20$ e $40$. Para o *stochastic backoff*
apenas a população com $40$ formigas é testada. A apresenta a média e o
desvio padrão para todos os experimentos. Nessa figura é possível notar
que a população gerada com $40$ formigas utilizando o *backoff* (sem a
parte probabilística) é a que apresenta o melhor desempenho. Isso se
deve ao fato de que o *backoff* faz uma busca exaustiva até encontrar
uma solução que passe por todos os nós, diferente do *stochastic
backoff* que pode interromper a busca sem visitar todos os nós. Embora o
resultado *backoff* seja melhor do que o da versão probabilística, ele
faz a suposição de que há um caminho que visite todos os nós, o que não
é realista. Independente da configuração utilizada, todos os parâmetros
geraram bons resultados, atingindo uma média próxima a $170$ e desvio
padrão menor do que $2.5$. Devidos as limitação presentes no *backoff*,
os demais testes são realizados utilizando o *stochastic backoff*, que
obteve um desempenho semelhante ao do *backoff* e é capaz de encontrar
soluções mesmo que não haja um caminho que percorre todos os nós, ou
seja, é mais robusto.

O próximo parâmetro a ser analisado é a taxa de decaimento do feromônio,
isto é, o quanto de feromônio presente na iteração anterior deve ser
mantido na próxima. O intervalo desse parâmetro é $[0, 1]$, portanto os
valores experimentados são $0.2$, $0.5$ e $0.8$. Quanto menor for o
valor desse parâmetro, maior é o impacto das iterações passadas na
atual. O ajuste desse parâmetro tem impacto na *exploration* e
*explotation* da colônia. A exibe os resultados dos experimentos. A taxa
de decaimento que gera o melhor resultado é $0.2$. Isso se deve ao fato
de que a colônia explora mais o espaço de solução, ao invés de confiar
apenas nos resultados obtidos préviamente. Como a taxa de decaimento
$0.2$ apresenta o melhor resultado, ele é mantido para os próximos
experimentos.

O parâmetro $\xi$ obteve o melhor resultado quando $\xi=0.2$, ou seja,
quando a probabilidade do *backoff* ocorrer é mais alto do que a dos
demais experimentos. A média e o desvio padrão final obtidos são
$172.6450$ e $1.0796$, respectivamente. O pior resultado ocorre quando
$\xi=0.5$, onde a média e o desvio padrão são $167.6450$ e $3.8267$,
respectivamente. A demonstra esses resultados.

A presença de elitismo é significativa. A população gerada sem elitismo
possui alta variância devido ao fato de não surgir rotas
probabilisticamente superior as demais. Com a presença do elitismo e,
por consequência, do reforço de feromônio na melhor rota encontrada até
o momento, a convergência é mais rápida e a variância se estabilaza mais
cedo. A demonstra esse efeito.

![Variação do tamanho da população](dataset_1_pop){width="\textwidth"}

[\[fig:ds1\_pop\]]{#fig:ds1_pop label="fig:ds1_pop"}

![Impacto do decaimento do
feromônio](dataset_1_decay){width="\textwidth"}

[\[fig:ds1\_decay\]]{#fig:ds1_decay label="fig:ds1_decay"}

![Impacto do coeficiente de decaimento
$\xi$](dataset_1_xi){width="\textwidth"}

[\[fig:ds1\_xi\]]{#fig:ds1_xi label="fig:ds1_xi"}

![Efeito do elitismo](dataset_1_e){width="\textwidth"}

[\[fig:ds1\_e\]]{#fig:ds1_e label="fig:ds1_e"}

Dataset 2
---------

Este dataset possui uma ordem de grandeza a mais do que o anterior. O
algoritmo *backoff* não é computacionalmente viável para esse tipo de
dataset, pois ele parte da premissa de que há um caminho que percorre
todas os nós uma única vez e ainda que haja, encontrá-lo pode ser muito
custoso em termos de tempo. Para minimizar esse problema foi introduzio
o *stochastic backoff*, que recua a solução encontrada
probabilisticamente e retorna a melhor solução encontrada durante as
tentativas. Outra implicação do tamanho do dataset é que devido ao
aumento na quantidade de vértices e arestas apenas 100 iterações são
executadas para cada experimento.

O primeiro teste verifica o impacto do tamanho da população na
eficiência do algoritmo. Os valores testados são $50$, $100$ e $200$. Os
resultados são apresentados na . O teste realizado com $200$ formigas
gerou soluções com custo até $987$ (o melhor resultado é $990$), porém o
desvio padrão se estabilizou em $4.3027$, enquanto que o teste realizado
com $50$ formigas obteve o desvio padrão igual a $3.2372$, o que sugere
que esta população convergiu primeiro do que aquela. O aumento no custo
médio obtido na população com $200$ formigas é apenas $0.003$ vezes
maior do que a menor, com $50$ formigas, portanto, o ganho não é
signicativo se comparado ao custo computacional para obtê-lo. A
população com $100$ formigas gerou o pior resultado, com desvio padrão
final de $5.1634$, o que sugere que o excesso de formigas aumenta a
variância das rotas encontradas e, portanto, leva mais tempo para
convergir do que aquelas com poucas formigas. Os próximos testes são
realizados com a população fixa em $50$ formigas.

O próximo parâmetro a ser verificado é o impacto do decaimento de
feromônio nos vértices. Os parâmetros verificados são: $0.2$, $0.5$ e
$0.8$. O progresso ao longo das iterações é representado na . O melhor
resultado obtido ocorre quando a taxa de decaimento é igual a $0.5$,
onde o custo médio após $100$ iterações é $978.3849$ com desvio padrão
igual a $3.2372$. Para o teste onde a taxa de decaimento é $0.2$, a
média é igual a $976.94$ com desvio padrão igual a $5.2338$. Já para o
teste com o parâmetro igual a $0.8$, a média é igual a $973.1480$ com
desvio padrão igual $3.9756$. O resultado indica que quando o decaimento
é muito baixo, ou seja, o feromônio em um vértice necessita de várias
iterações para decair, há uma lentidão na convergência na população e
isso é indicado pela magnitude do desvio padrão para a população gerada
com coeficiente de decaimento igual a $0.2$. Para os demais testes, a
taxa de decaimento de feromônio é fixada em $0.5$.

Outro parâmetro analisado é o coeficiente de decaimento do *backoff*,
que controla a probabilidade da formiga recuar e tentar encontrar outra
solução quando não há vértices para nós ainda não visitados. Os valores
experimentados são: $0.2$, $0.5$ e $0.8$. Os resultados obtidos são
representados na . Os resultados indicam que a configuração com o $\xi$
igual a $0.5$ alcança o melhor eficiência. O custo máximo médio é
$982.40$ e o custo médio é $978.4080$ com desvio padrão igual a
$2.9823$. Para as demais configurações, o desvio padrão final é superior
a $5$, o que indica que a desistência prematura no *backoff* ou o
excesso de tentativas levam ao aumento da variância entre as iterações,
sem ganho nas demais estatísticas. Para o próximo experimento o valor de
$\xi$ é mantido fixado em $0.5$.

Por fim, o último parâmetro verificado é a presença ou ausência de
elitismo. O mecânismo de elitismo no ACO é implementado aplicando um
reforço na melhor rota encontrada até a iteração mais recente. O
elitismo tem impacto na convergência do algoritmo, o que é indicado pela
variância final das soluções geradas pelo algoritmo. O desvio padrão com
a presença de elitismo é aproximadamente $3$ vezes maior do que sem.
Portanto, a presença de elitismo influencia significativa na
convergência do algoritmo sob as configurações desenvolvidas. A
representa o resultado das colônias com e sem elitismo ao longo de $100$
iterações.

[\[fig:dataset\_2\]]{#fig:dataset_2 label="fig:dataset_2"}

![Variação do tamanho da população](dataset_2_pop){width="\textwidth"}

[\[fig:ds2\_pop\]]{#fig:ds2_pop label="fig:ds2_pop"}

![Impacto do decaimento do
feromônio](dataset_2_decay){width="\textwidth"}

[\[fig:ds2\_decay\]]{#fig:ds2_decay label="fig:ds2_decay"}

![Impacto do coeficiente de
decaimento](dataset_2_xi){width="\textwidth"}

[\[fig:ds2\_xi\]]{#fig:ds2_xi label="fig:ds2_xi"}

![Efeito do elitismo](dataset_2_e){width="\textwidth"}

[\[fig:ds2\_e\]]{#fig:ds2_e label="fig:ds2_e"}

Dataset 3
---------

Este dataset possui uma ordem de grandeza em relação ao anterior e o
custo computacional de verificar o impacto de cada parâmetro é
impraticável, portanto, neste dataset o objetivo é verificar a
performance do algoritmo em termos de escalabilidade.

O ACO com *stochastic backoff* foi capaz de encontrar soluções com custo
máximo igual a $9756.40$, porém o tempo de processamente necessário foi
muito elevado e, portanto, não foi possível realizar o *tuning* dos
parâmetros devido a limitações de tempo. Isso sugere a necessidade de um
novo design do algoritmo em favor da escalabilidade.

Os resultados da nova implementação do algoritmo ACO foi projetada com
uso da álgebra linear. A discrepância entre as implementações é
exorbitante. Enquanto a primeira implementação não foi capaz de
encontrar soluções em 230H, a nova foi capaz de gerar uma população
quatro vezes maior (200 formigas) em apenas 40min no mesmo hardware.
Além do aumento significativo de desempenho, a qualidade da população
foi significativamente maior. Na as populações com 100 ou mais
indivíduos foram gerados pela nova implementação e os resultados indicam
claramente que a quantidade de iterações até alcançar um ponto de
convergência (um máximo local[^2]) é relativamente rápido.

![Variação do tamanho da população](dataset_3_pop){width="\textwidth"}

[\[fig:ds3\_pop\]]{#fig:ds3_pop label="fig:ds3_pop"}

![Impacto do coeficiente de
decaimento](dataset_3_decay){width="\textwidth"}

[\[fig:ds3\_decay\]]{#fig:ds3_decay label="fig:ds3_decay"}

![Impacto da presença de elitismo](dataset_3_pop){width="\textwidth"}

[\[fig:ds3\_elitism\]]{#fig:ds3_elitism label="fig:ds3_elitism"}

Conclusão {#sec:conclusion}
=========

Os resultados indicam que a presença do *stochastic backoff* é
significativo e o único capaz de explorar a grafos muito densos. O
*backoff*, em sua forma inicial, depende do conhecimento de determinadas
informações sobre os grafos, que nem sempre são verificáveis, como em
problemas NP-*complete* e para problemas combinatoriais com muitos
níveis. Além disso, os resultados gerados demonstram que o algoritmo
possui bom convergência, pois independente dos parâmetros utilizandos,
sempre houve uma melhoria em relação ao estado inicial.

O tamanho da população traz consigo o aumento da variância dos
resultados, porém isso é reduzido com ao longo das iterações. Com
populações consideralvente grandes a variância volta a diminuir,
indicando que a variância pode ser modelada em uma função parabólica,
porém mais testes são necessários para comprovar este efeito. Outro
fator interessante a partir de 20 formigas, os ganhos de populações
maiores não são tão siginificativos, embora haja a melhoria. O
coeficiente de decaimento do *backoff* aumenta a performance do
algoritmo e o torna mais robusto. O uso de arquivos garante que não há
perda de boas soluções encontradas durante os *backoffs*. O uso de
formigas elitistas também se mostrou impactante na convergência da
população pelo reforço de feromônio na melhor rota encontrada até a
iteração atual. Por fim, o coeficiente de decaimento atinge melhores
resultados para valores pequenos (menores do que $0.5$), o que sugere
que o acumulo de resultados obtidos influencia positivamente o
desempenho da colônia.

Outro fator interessante é que ambos os algoritmos alcançam valores
próximos do máximo possível (informação dada para as duas primeiras
redes). O desvio padrão baixo também é outro indício que os algoritmos
são estáveis, pois mesmo com o fator aleatório, eles são capazes de
convergir para soluções próximas das ótimas.

Um problema identificado é a falta de escalabilidade. O algoritmo gastou
mais de 200 horas (isso depende fortemente do hardware) para processar o
dataset 3. A análise do algoritmo indica que o trecho que demanda mais
tempo é a construção da rota, mais especificamente, a decisão de qual
vértice tomar. Uma possível solução é transformar o problema para forma
matricial e utilizar álgebra linear para aproveitar características
específicas do *hardware*, como *SIMD instructions*, e aumentar a
performance em processadores super-escalares, uma vez que, a presença de
saltos no algoritmo é reduzido e que permite a execução especulativa de
instruções.

O novo design resolveu o problema descrito acima, porém a complexidade
de implementar o backoff utilizando a algebra linear pode reduzir a
performance do algoritmo novamente. A solução é reformular o *backoff*
para um modelo baseado em álgebra linear. Portanto, a conclusão sobre o
*backoff* com base nos resultados obtidos utilizando-se o *backoff*
($\xi=0$), *stochastic backoff* e a ausência de *backoff*
($\xi=+\infty$) é que ele não traz ganhos significativos nos testes
realizados. Os ganhos do uso desse mecânismo deve surgir na aplicação em
grafos mais complexos, onde a densidade de vértices é menor e há muitos
nós isolados (1 vizinho, por exemplo).

Para trabalhos futuros há muitas possibilidades. Alguma heurística pode
ser utilizada para gerar uma solução que agrege, sempre que possível, as
soluções criadas pelas formigas na iteração atual ou combinar os boas
soluções anteriores.

[^1]: matheus.candido\@dcc.ufmg.br

[^2]: Possivelmente um máximo global?
