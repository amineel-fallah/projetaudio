Add-Type -AssemblyName System.Speech

$phrases = @{
    "test_audio_happy.wav" = "I am so happy today. This is wonderful news."
    "test_audio_sad.wav" = "I feel so sad and lonely. Everything is going wrong."
    "test_audio_angry.wav" = "This is unacceptable. I am furious. How dare you."
    "test_audio_neutral.wav" = "The weather today is cloudy. The temperature is fifteen degrees."
    "test_audio_fearful.wav" = "Oh no what was that noise. I am scared. Please help me."
    "test_audio_surprised.wav" = "Oh my goodness. I cannot believe this. What a surprise."
}

foreach ($file in $phrases.Keys) {
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $synth.SetOutputToWaveFile($file)
    $synth.Speak($phrases[$file])
    $synth.Dispose()
    Write-Host "Created: $file"
}

Write-Host "All audio files created!"
