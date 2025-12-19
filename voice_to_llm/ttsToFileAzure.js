import os from "os";
import path from "path";

export async function ttsToFileAzure(text, voice = "en-US-AriaNeural") {
    const sdk = (await import("microsoft-cognitiveservices-speech-sdk")).default;
    const out = path.join(os.tmpdir(), `tts_${Date.now()}.wav`);
    const key = process.env.SPEECH_KEY;
    const region = process.env.SPEECH_REGION;
    if (!key || !region)
        throw new Error("SPEECH_KEY and SPEECH_REGION must be set");

    const speechCfg = sdk.SpeechConfig.fromSubscription(key, region);
    speechCfg.speechSynthesisVoiceName = voice;

    const audioCfg = sdk.AudioConfig.fromAudioFileOutput(out);
    const synthesizer = new sdk.SpeechSynthesizer(speechCfg, audioCfg);

    await new Promise((resolve, reject) =>
        synthesizer.speakTextAsync(
        text,
        () => {
            synthesizer.close();
            resolve();
        },
        (e) => {
            synthesizer.close();
            reject(e);
        }
        )
    );
    return out;
}
