# Dataset Notes

## Bot-IoT (training — known botnet families)
- Source: UNSW Canberra Cyber Range Lab
- Use the extracted "5% subset" (not the full ~72GB capture) for feasible local/Kaggle use
- Key columns: flow identifiers, timing, volume, TCP flags, statistical features, category/subcategory labels

## N-BaIoT (zero-day test — unseen families)
- Source: real traffic from infected commercial IoT devices (Mirai, BASHLITE families)
- Different collection methodology than Bot-IoT — used here specifically for genuine cross-dataset generalization testing

## Notes on feature alignment
Bot-IoT and N-BaIoT do not share an identical feature schema. Alignment
strategy and column-mapping decisions should be documented here as they're
made, so the preprocessing step is reproducible.

## Download links
_(fill in once finalized)_