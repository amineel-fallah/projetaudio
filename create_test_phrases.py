#!/usr/bin/env python3
"""
Script pour créer un guide de phrases à enregistrer
"""

print("=" * 70)
print("  🎤 Guide d'Enregistrement - Phrases de Test")
print("=" * 70)

phrases = {
    "😐 NEUTRAL": [
        "Bonjour, comment allez-vous aujourd'hui ?",
        "Je vais à la bibliothèque cet après-midi.",
        "Il est treize heures et quinze minutes.",
        "Le temps est nuageux ce matin.",
        "Je dois terminer ce rapport."
    ],
    "😊 HAPPY": [
        "C'est fantastique ! Je suis super content !",
        "J'ai réussi mon examen ! Quelle joie !",
        "Nous partons en vacances demain ! Génial !",
        "J'adore cette musique, elle est magnifique !",
        "Quel beau cadeau, merci beaucoup !"
    ],
    "😢 SAD": [
        "C'est vraiment dommage... je suis déçu.",
        "Je me sens seul aujourd'hui.",
        "Cette nouvelle m'a vraiment attristé.",
        "Je n'arrive pas à surmonter cette épreuve.",
        "Tout semble si difficile en ce moment."
    ],
    "😡 ANGRY": [
        "C'est inacceptable ! Je ne suis pas d'accord !",
        "J'en ai assez de cette situation !",
        "Comment osez-vous me parler ainsi ?!",
        "Cela me met hors de moi !",
        "Je refuse catégoriquement cette décision !"
    ],
    "😨 FEARFUL": [
        "Oh non... j'ai peur que ça arrive...",
        "Je ne sais pas si je vais y arriver...",
        "Cela m'inquiète beaucoup, vraiment beaucoup.",
        "J'ai un mauvais pressentiment à ce sujet.",
        "Je tremble rien que d'y penser."
    ],
    "😲 SURPRISED": [
        "Quoi ?! Vraiment ?! C'est incroyable !",
        "Je n'en crois pas mes oreilles !",
        "Wow ! Je ne m'attendais pas à ça !",
        "C'est une surprise totale !",
        "Ça alors ! Quelle nouvelle !"
    ]
}

conseils = {
    "😐 NEUTRAL": "Parlez de manière monotone, sans émotion particulière.",
    "😊 HAPPY": "Souriez en parlant, voix enjouée et dynamique.",
    "😢 SAD": "Voix basse, lente, sans énergie.",
    "😡 ANGRY": "Voix forte, tendue, articulation marquée.",
    "😨 FEARFUL": "Voix tremblante, hésitante, anxieuse.",
    "😲 SURPRISED": "Voix haute, exclamative, dynamique."
}

print("\n🎭 30 PHRASES À ENREGISTRER (5 par émotion)\n")
print("📝 Instructions:")
print("   1. Lancer: python app.py")
print("   2. Cliquer sur le bouton micro 🎙️")
print("   3. Lire UNE phrase ci-dessous (3-5 secondes)")
print("   4. Observer le résultat")
print("   5. Noter la précision\n")
print("-" * 70)

for emotion, phrase_list in phrases.items():
    print(f"\n{emotion}")
    print(f"💡 Conseil: {conseils[emotion]}")
    print()
    for i, phrase in enumerate(phrase_list, 1):
        print(f"   {i}. \"{phrase}\"")

print("\n" + "=" * 70)
print("  📊 ÉVALUATION")
print("=" * 70)
print("""
Après avoir testé toutes les phrases, calculez:

Précision = (Nombre correct / Total testé) × 100%

Exemple:
  - Testé: 30 phrases
  - Correct: 23
  - Précision: 76.7%

Si précision > 70% → Excellent ! ✅
Si précision 50-70% → Bon 👍
Si précision < 50% → À améliorer 🔧
""")

print("💾 Sauvegarder ce fichier:")
with open("test_audios/PHRASES_TEST.txt", "w", encoding="utf-8") as f:
    f.write("🎤 PHRASES DE TEST - RECONNAISSANCE D'ÉMOTIONS\n")
    f.write("=" * 60 + "\n\n")
    
    for emotion, phrase_list in phrases.items():
        f.write(f"{emotion}\n")
        f.write(f"Conseil: {conseils[emotion]}\n\n")
        for i, phrase in enumerate(phrase_list, 1):
            f.write(f"  {i}. {phrase}\n")
        f.write("\n")
    
    f.write("\nInstructions:\n")
    f.write("1. Lancer: python app.py\n")
    f.write("2. Cliquer micro 🎙️\n")
    f.write("3. Lire une phrase (3-5s)\n")
    f.write("4. Noter le résultat\n")
    f.write("5. Répéter pour toutes les phrases\n")

print("✅ Phrases sauvegardées dans: test_audios/PHRASES_TEST.txt")
print("\n🚀 Prêt à tester ! Lancez: python app.py")
