# Hardware de investimento mínimo (Fase 0)

Princípio: gastar em **dados**, não em sensor. O que diferencia o produto é o dataset
pareado Nelore e a experiência no celular; a câmera do ponto fixo pode ser usada e barata
até a Fase 3.

## Kit A — celular do próprio produtor (R$ 0)

- Qualquer celular com câmera. iPhone Pro (LiDAR) melhora a escala, mas não é obrigatório.
- **Marcador de escala**: uma folha A4 (0,297 × 0,210 m) plastificada, ou placa de 30 × 30 cm
  com xadrez preto e branco, encostada no animal ou na cerca à mesma distância. O importador
  usa `marker_px/marker_m`; o app da Fase 2 detecta o marcador automaticamente.
- Formulário de coleta offline: `tools/coleta.html` (abra no celular, funciona sem internet,
  exporta CSV). Importação: `bo import-phone`.
- Resultado esperado (validado em dataset público CC BY): **sem escala o celular não funciona
  (erro 17%, pior que a média); com escala, 7 a 8% com uma foto** e segmentador genérico.
  Vídeo (dezenas de frames) e modelo treinado em Nelore devem levar para 5 a 6%.

## Kit B — ponto fixo com Kinect v2 usado (R$ 600 a 1.100)

| item | preço ref. (BR, usado/novo) | obs. |
|---|---|---|
| Kinect v2 (Xbox One) usado | R$ 300 a 600 | ToF 512×424, o MESMO sensor dos datasets públicos que validamos |
| Adaptador Kinect → USB 3 / fonte | R$ 100 a 250 | original ou compatível; exige porta USB 3 |
| Notebook antigo com USB 3 (i5, 8 GB) | R$ 0 (reaproveitado) ou R$ 600 usado | roda `bo capture --source kinect2`; Linux com libfreenect2 |
| Suporte: cano galvanizado + abraçadeiras + caixa plástica com visor | R$ 100 a 200 | altura 3,0 a 3,5 m sobre a passagem, cobertura contra sol e chuva |
| Nobreak pequeno (opcional) | R$ 250 | quedas de energia corrompem gravação |
| Internet | celular como roteador ou pendrive | o daemon grava offline e enfileira |

Alternativa ainda mais barata: **Kinect v1 (Xbox 360)** usado (R$ 100 a 200) + adaptador; roda
via OpenNI2 até em Raspberry Pi 4 (`--source openni2`). Resolução menor (640×480, menos
precisa em profundidade), suficiente para o piloto de coleta.

Limitações aceitas na Fase 0: sensores IR saturam com sol direto (instalar sob cobertura e
capturar de manhã, que é também o horário do protocolo de pesagem) e não são IP65 (caixa
plástica). Orbbec Femto Bolt / OAK-D Pro PoE / Jetson só entram quando houver cliente pagando
(Fase 3), com a instalação padronizada.

## Instalação do software (Kit B, Ubuntu 22.04/24.04)

```bash
sudo apt install -y build-essential cmake libusb-1.0-0-dev libturbojpeg0-dev libglfw3-dev
git clone https://github.com/OpenKinect/libfreenect2 && cd libfreenect2 && mkdir build && cd build
cmake .. -DENABLE_CXX11=ON && make -j && sudo make install
sudo cp ../platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/ && sudo udevadm control --reload
pip install pylibfreenect2
bo capture --source kinect2 --out ~/bo_capture --site fazenda_serra_balanca
```

Sem hardware, tudo roda com `--source replay:<pasta>` sobre PNGs gravados.

## Ponto de instalação recomendado na Fazenda Serra

1. **Saída da balança** (primeiro): cada passagem já vem com o peso → dataset de treino
   sem trabalho extra. Câmera dorsal a ~3,2 m, olhando para baixo, animal enquadrado inteiro.
2. **Bebedouro** (depois): monitoramento contínuo, GMD semanal. Exige RFID para identificar.

Registro da balança: se ela tiver saída serial/Bluetooth, gravamos direto; se não, o operador
digita brinco + peso no `tools/coleta.html` e o `bo pair` casa por horário (janela de 90 s).
