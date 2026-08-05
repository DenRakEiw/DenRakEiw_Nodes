# Multi-Image Aspect Ratio Composer

Eine fortschrittliche ComfyUI Node zum Kombinieren mehrerer Bilder in ein einziges Ausgabebild mit spezifischem Seitenverhältnis.

## Features

### 🎛️ Dynamische Input-Kontrolle
- **Input Count Selector**: Wählen Sie zwischen 1-8 Input-Bildern
- **Update Button**: Aktualisiert die Node-Inputs dynamisch
- **Automatische UI-Anpassung**: Die Benutzeroberfläche passt sich automatisch an die gewählte Anzahl an

### 📐 Seitenverhältnis-Presets
- **1:1 (Square)**: Quadratisches Format
- **4:3 (Standard)**: Klassisches Foto-Format
- **3:4 (Portrait)**: Hochformat
- **16:9 (Widescreen)**: Breitbild-Format
- **9:16 (Vertical)**: Vertikales Breitbild
- **21:9 (Ultrawide)**: Ultra-Breitbild
- **9:21 (Ultra Vertical)**: Ultra-Vertikal
- **3:2 (Photo)**: Standard-Foto-Format
- **2:3 (Photo Portrait)**: Foto-Hochformat
- **5:4 (Classic)**: Klassisches Format
- **4:5 (Classic Portrait)**: Klassisches Hochformat
- **16:10 (Monitor)**: Monitor-Format
- **10:16 (Monitor Portrait)**: Monitor-Hochformat
- **2:1 (Panorama)**: Panorama-Format
- **1:2 (Vertical Panorama)**: Vertikales Panorama

### 🎯 Megapixel-Auswahl
- **0.5 MP bis 32 MP**: Verschiedene Auflösungsoptionen
- **Automatische Berechnung**: Breite und Höhe werden automatisch berechnet
- **64-Pixel-Teilbarkeit**: Alle Ausgabedimensionen sind durch 64 teilbar

### 🎨 Anordnungsoptionen
- **Horizontal**: Bilder nebeneinander
- **Vertical**: Bilder übereinander
- **Smart Grid**: Intelligentes Raster-Layout (NEU!)
  - Optimiert für das gewählte Seitenverhältnis
  - Verwendet ALLE Bilder (keine werden mehr "verloren")
  - Flexible Zeilen mit unterschiedlicher Bildanzahl
- **Classic Grid**: Traditionelles starres Raster-Layout

### ⚙️ Erweiterte Optionen
- **Spacing**: Abstand zwischen Bildern (0-100 Pixel)
- **Background Color**: Hintergrundfarbe (Schwarz, Weiß, Transparent)
- **Automatisches Skalieren**: Bilder werden automatisch skaliert und zentriert beschnitten

### 🎭 Face Detection (NEU!)
- **Face Detection**: Ein/Aus-Schalter für intelligentes Gesichtserkennung-Cropping
- **Haar Cascade**: Schnelle Gesichtserkennung mit OpenCV
- **DNN Face**: Erweiterte Deep Learning Gesichtserkennung (falls verfügbar)
- **Confidence**: Einstellbare Erkennungsgenauigkeit (1.1 - 3.0)
- **Intelligentes Cropping**: Bilder werden um erkannte Gesichter zentriert
- **Fallback**: Automatischer Rückfall auf Center-Crop wenn keine Gesichter erkannt

## Verwendung

### Grundlegende Schritte:
1. **Input Count einstellen**: Wählen Sie die gewünschte Anzahl von Input-Bildern (1-8)
2. **Update Inputs klicken**: Aktualisiert die Node mit der entsprechenden Anzahl von Bild-Inputs
3. **Aspect Ratio wählen**: Wählen Sie das gewünschte Seitenverhältnis
4. **Megapixels einstellen**: Bestimmen Sie die Zielauflösung
5. **Arrangement wählen**: Horizontal, Vertikal, Smart Grid oder Classic Grid
6. **Face Detection konfigurieren**:
   - **Disabled**: Standard Center-Cropping
   - **Haar Cascade**: Schnelle Gesichtserkennung
   - **DNN Face**: Erweiterte Gesichtserkennung
7. **Confidence anpassen**: Erkennungsgenauigkeit (höher = strenger)
8. **Bilder verbinden**: Verbinden Sie Ihre Bilder mit den Input-Slots
9. **Ausführen**: Die Node erstellt das komponierte Bild

### Ausgaben:
- **composed_image**: Das finale komponierte Bild
- **width**: Breite des Ausgabebildes
- **height**: Höhe des Ausgabebildes
- **info**: Informationsstring mit Details zur Komposition

## Technische Details

### Bildverarbeitung:
- **Intelligentes Cropping**: Bilder werden um erkannte Gesichter oder zentriert beschnitten
- **Face Detection**: OpenCV-basierte Gesichtserkennung für optimales Cropping
- **Bilineare Interpolation**: Hochwertige Skalierung der Bilder
- **Automatische Größenanpassung**: Jedes Bild wird optimal in den verfügbaren Raum eingepasst

### Face Detection Details:
- **Haar Cascade**:
  - Schnelle, CPU-effiziente Gesichtserkennung
  - Gut für Frontalansichten
  - Confidence 1.1-1.5 empfohlen
- **DNN Face**:
  - Erweiterte Deep Learning Erkennung
  - Bessere Genauigkeit bei verschiedenen Winkeln
  - Etwas langsamer als Haar Cascade
- **Multiple Faces**:
  - Bei mehreren Gesichtern wird das größte verwendet
  - Fallback auf Center-Crop wenn keine Gesichter erkannt
- **Debug Output**:
  - Konsolen-Ausgabe zeigt erkannte Gesichter
  - Hilfreich für Troubleshooting

### Smart Grid Algorithmus (NEU!):
Der intelligente Grid-Algorithmus optimiert die Anordnung basierend auf:
- **Ziel-Seitenverhältnis**: Berechnet optimale Zeilen/Spalten-Verteilung
- **Alle Bilder verwenden**: Keine Bilder gehen mehr verloren
- **Flexible Layouts**: Verschiedene Bildanzahl pro Zeile

**Beispiele für 8 Bilder:**
- **16:9 Ziel**: Layout [4, 4] (2 Zeilen mit je 4 Bildern)
- **1:1 Ziel**: Layout [3, 3, 2] (3 Zeilen: 3+3+2 Bilder)
- **9:16 Ziel**: Layout [2, 2, 2, 2] (4 Zeilen mit je 2 Bildern)

### Classic Grid Layout:
- **1 Bild**: 1x1 Raster
- **2 Bilder**: 2x1 Raster
- **3-4 Bilder**: 2x2 Raster
- **5-6 Bilder**: 3x2 Raster
- **7-8 Bilder**: 4x2 Raster (⚠️ kann Bilder "verlieren")

### Dimensionsberechnung:
```python
# Beispiel für 16:9 bei 2 MP:
total_pixels = 2_000_000
ratio = 16/9
height = sqrt(total_pixels / ratio)
width = height * ratio
# Rundung auf nächste 64er-Grenze
width = round(width / 64) * 64
height = round(height / 64) * 64
```

## Beispiele

### Horizontal Layout:
- 3 Bilder nebeneinander
- 16:9 Seitenverhältnis
- 4 MP Auflösung
- Ergebnis: 2560x1440 Pixel

### Grid Layout:
- 4 Bilder in 2x2 Anordnung
- 1:1 Seitenverhältnis
- 8 MP Auflösung
- Ergebnis: 2816x2816 Pixel

### Vertical Layout:
- 2 Bilder übereinander
- 9:16 Seitenverhältnis
- 2 MP Auflösung
- Ergebnis: 1088x1920 Pixel

### Portrait Composition mit Face Detection:
- 4 Portrait-Bilder in 2x2 Grid
- 1:1 Seitenverhältnis
- Face Detection: Haar Cascade
- Confidence: 1.3
- Ergebnis: Alle Gesichter optimal zentriert

## Tipps

1. **Optimale Bildqualität**: Verwenden Sie Bilder mit ähnlicher Auflösung für beste Ergebnisse
2. **Spacing nutzen**: Fügen Sie Abstand zwischen Bildern hinzu für bessere Trennung
3. **Grid für viele Bilder**: Bei 4+ Bildern ist das Grid-Layout oft am besten
4. **Megapixel anpassen**: Höhere MP-Werte für bessere Qualität, niedrigere für Performance
5. **Seitenverhältnis beachten**: Wählen Sie das Seitenverhältnis passend zu Ihrem Verwendungszweck
6. **Face Detection für Portraits**: Aktivieren Sie Face Detection bei Portrait-Bildern
7. **Confidence anpassen**: Niedrigere Werte (1.1-1.3) für mehr Erkennungen, höhere (1.5-2.0) für genauere
8. **Performance**: Face Detection kann die Verarbeitung verlangsamen - bei Bedarf deaktivieren
9. **Debugging**: Konsolen-Output zeigt erkannte Gesichter für Troubleshooting

## Kompatibilität

- **ComfyUI**: Vollständig kompatibel
- **Torch**: Nutzt PyTorch für Bildverarbeitung
- **Memory**: Optimiert für verschiedene Speichergrößen
- **Batch Processing**: Unterstützt Batch-Verarbeitung

## Fehlerbehebung

### Häufige Probleme:
1. **Keine Bilder sichtbar**: Überprüfen Sie, ob alle gewünschten Bild-Inputs verbunden sind
2. **Falsche Dimensionen**: Klicken Sie "Update Inputs" nach Änderung der Input Count
3. **Speicherfehler**: Reduzieren Sie die Megapixel-Einstellung
4. **Qualitätsverlust**: Erhöhen Sie die Megapixel-Einstellung oder verwenden Sie weniger Bilder

### Debug-Informationen:
Die Node gibt detaillierte Debug-Informationen in der Konsole aus:
- Zieldimensionen
- Seitenverhältnis
- Anzahl verarbeiteter Bilder
- Finale Ausgabedimensionen
