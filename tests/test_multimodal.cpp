#include "oil/multimodal.h"
#include "oil/model.h"
#include "oil/tensor.h"
#include "oil/math.h"
#include "oil/types.h"
#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include <string>

using namespace oil;

static int g_tests = 0;
static int g_passed = 0;

#define CHECK(cond, msg) do { \
    g_tests++; \
    if (!(cond)) { printf("  FAIL: %s\n", msg); } \
    else { g_passed++; printf("  PASS: %s\n", msg); } \
} while(0)

#define CHECK_CLOSE(a, b, eps, msg) CHECK(std::fabs((a)-(b)) < (eps), msg)

static void test_cross_modal_attention() {
    printf("\n=== H1: CrossModalAttention ===\n");
    CrossModalAttention cma(32);
    std::vector<Tensor> modalities;
    modalities.push_back(Tensor({32}));
    modalities.push_back(Tensor({32}));
    modalities.push_back(Tensor({32}));
    modalities[0].fill(1.0f);
    modalities[1].fill(2.0f);
    modalities[2].fill(3.0f);

    auto out = cma.forward(modalities);
    CHECK(out.numel() == 32, "output shape matches hidden dim");

    auto out_empty = cma.forward({});
    CHECK(out_empty.numel() == 32, "empty modalities returns default tensor");

    auto out_single = cma.forward({modalities[0]});
    CHECK(out_single.numel() == 32, "single modality works");
}

static void test_joint_multimodal_model() {
    printf("\n=== H2: JointMultimodalModel ===\n");
    JointMultimodalModel jmm(nullptr, 64);
    Tensor text({1, 10, 64}), image({1, 8, 64}), audio({1, 5, 64});
    text.fill(1.0f); image.fill(2.0f); audio.fill(3.0f);
    auto out = jmm.forward(text, image, audio);
    CHECK(out.numel() > 0, "joint model produces output");
    CHECK(std::isfinite(out.data<float>()[0]), "output values are finite");
}

static void test_imagenet_classifier() {
    printf("\n=== H3: ImageNetClassifier ===\n");
    ImageNetClassifier inc(nullptr, 100);
    Tensor image({2, 768});
    image.fill(0.5f);
    auto logits = inc.classify(image);
    CHECK(logits.dim(0) == 2, "classifier outputs batch dim");
    CHECK(logits.dim(1) == 100, "classifier outputs 100 classes");

    for (int64_t i = 0; i < logits.numel(); i++)
        CHECK(std::isfinite(logits.data<float>()[i]), "classifier outputs are finite");
}

static void test_speech_recognizer() {
    printf("\n=== H4: SpeechRecognizer ===\n");
    SpeechRecognizer sr(nullptr);
    Tensor audio({1, 100, 40});
    audio.fill(0.5f);
    auto text = sr.transcribe(audio);
    CHECK(!text.empty(), "transcribe returns non-empty text");
}

static void test_ocr_pipeline() {
    printf("\n=== H5: OCRPipeline ===\n");
    OCRPipeline ocr(nullptr);
    Tensor image({1, 50, 256});
    image.fill(0.3f);
    auto text = ocr.recognize(image);
    CHECK(!text.empty(), "OCR returns non-empty text");
}

static void test_video_understanding() {
    printf("\n=== H6: VideoUnderstanding ===\n");
    VideoUnderstanding vu(nullptr);
    Tensor frames({1, 16, 768});
    frames.fill(0.5f);
    auto desc = vu.describe(frames);
    CHECK(!desc.empty(), "video description non-empty");
    CHECK(desc.find("[video") != std::string::npos, "description contains marker");
}

static void test_image_captioning() {
    printf("\n=== H7: ImageCaptioning ===\n");
    ImageCaptioning ic(nullptr, nullptr);
    Tensor image({1, 768});
    image.fill(1.0f);
    auto cap = ic.caption(image, 5);
    CHECK(!cap.empty(), "caption returns text");
}

static void test_visual_qa() {
    printf("\n=== H8: VisualQA ===\n");
    VisualQA vqa(nullptr, nullptr);
    Tensor image({1, 768});
    image.fill(1.0f);
    auto ans = vqa.answer(image, "what is this?");
    CHECK(!ans.empty(), "VQA returns answer");
}

static void test_text_to_image() {
    printf("\n=== H9: TextToImage ===\n");
    TextToImage t2i(64, 256);
    auto img = t2i.generate("a cat", 10);
    CHECK(img.numel() > 0, "text-to-image produces output");
    CHECK(img.dim(0) == 3, "image has 3 channels");
    CHECK(img.dim(1) == 256, "image height");
    CHECK(img.dim(2) == 256, "image width");
}

static void test_audio_synthesizer() {
    printf("\n=== H10: AudioSynthesizer ===\n");
    AudioSynthesizer as;
    Tensor mel({80, 50});
    mel.fill(0.5f);
    auto wav = as.synthesize(mel);
    CHECK(wav.numel() > 0, "synthesizer produces waveform");
    for (int64_t i = 0; i < wav.numel(); i++)
        CHECK(std::isfinite(wav.data<float>()[i]), "waveform values finite");
}

static void test_mel_spectrogram() {
    printf("\n=== H11: MelSpectrogram ===\n");
    MelSpectrogram ms(22050, 80);
    Tensor waveform({22050}); // 1 second
    for (int64_t i = 0; i < 22050; i++)
        waveform.data<float>()[i] = std::sin(2.0f * 3.14159f * 440.0f * i / 22050.0f);
    auto mel = ms.compute(waveform);
    CHECK(mel.dim(0) == 80, "mel spectrogram has 80 bands");
    CHECK(mel.dim(1) > 0, "mel spectrogram has frames");
    for (int64_t i = 0; i < mel.numel(); i++)
        CHECK(std::isfinite(mel.data<float>()[i]), "mel values finite");
}

static void test_audio_feature_extractor() {
    printf("\n=== H12: AudioFeatureExtractor ===\n");
    AudioFeatureExtractor afe(nullptr);
    Tensor audio({1, 1024});
    audio.fill(0.5f);
    auto feat = afe.extract(audio);
    CHECK(feat.numel() > 0, "extractor produces features");
}

static void test_multimodal_tokenizer() {
    printf("\n=== H13: MultiModalTokenizer ===\n");
    MultiModalTokenizer mmt;
    auto tokens = mmt.encode("hello world");
    CHECK(!tokens.empty(), "tokenizer encodes text");
    CHECK(mmt.vocab_size() == 32000, "default vocab size");
    CHECK(mmt.image_token_id() == 30000, "image token id");
    CHECK(mmt.audio_token_id() == 31000, "audio token id");

    auto decoded = mmt.decode(tokens);
    CHECK(!decoded.empty(), "tokenizer decodes text");

    auto img_tokens = mmt.encode_image(Tensor({10}));
    CHECK(!img_tokens.empty(), "image encoding works");
    CHECK(img_tokens[0] == 30000, "image encoding returns image token");

    auto aud_tokens = mmt.encode_audio(Tensor({10}));
    CHECK(!aud_tokens.empty(), "audio encoding works");
    CHECK(aud_tokens[0] == 31000, "audio encoding returns audio token");
}

static void test_modality_encoder() {
    printf("\n=== H14: ModalityEncoder ===\n");
    ModalityEncoder me(nullptr, "text");
    Tensor inp({1, 10});
    inp.fill(1.0f);
    auto encoded = me.encode(inp);
    CHECK(encoded.numel() > 0, "modality encoder works");
}

static void test_cross_modal_alignment() {
    printf("\n=== H15: CrossModalAlignment ===\n");
    CrossModalAlignment cma(nullptr, nullptr, 0.07f);
    int64_t B=4, D=16;
    Tensor img_emb({B, D}), txt_emb({B, D});
    img_emb.fill(1.0f);
    txt_emb.fill(1.0f);
    float loss = cma.contrastive_loss(img_emb, txt_emb);
    CHECK(loss > 0, "contrastive loss positive");
    CHECK(std::isfinite(loss), "contrastive loss finite");

    auto aligned = cma.align(img_emb, txt_emb);
    CHECK(aligned.dim(0) == B, "aligned output batch dim");
    CHECK(aligned.dim(1) == D, "aligned output feature dim");
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("MYTHOS.cpp — Multi-Modal (H1-H15) Test Suite\n");
    printf("============================================\n");

    test_cross_modal_attention();
    test_joint_multimodal_model();
    test_imagenet_classifier();
    test_speech_recognizer();
    test_ocr_pipeline();
    test_video_understanding();
    test_image_captioning();
    test_visual_qa();
    test_text_to_image();
    test_audio_synthesizer();
    test_mel_spectrogram();
    test_audio_feature_extractor();
    test_multimodal_tokenizer();
    test_modality_encoder();
    test_cross_modal_alignment();

    printf("\n============================================\n");
    printf("Results: %d / %d tests passed", g_passed, g_tests);
    if (g_passed == g_tests) printf(" -- ALL PASSED\n");
    else printf(" (%d FAILED)\n", g_tests - g_passed);
    return (g_passed == g_tests) ? 0 : 1;
}
