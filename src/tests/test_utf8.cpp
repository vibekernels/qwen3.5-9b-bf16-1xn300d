// Tests for UTF-8 streaming buffer logic used in generate().
// This extracts the exact buffering algorithm from engine.cu so we can
// test it without a GPU or model.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;
static int g_tests = 0;

#define EXPECT_EQ_STR(a, b) do { \
    g_tests++; \
    if ((a) != (b)) { \
        fprintf(stderr, "FAIL %s:%d: %s != %s\n  got: \"%s\" vs \"%s\"\n", \
                __FILE__, __LINE__, #a, #b, \
                std::string(a).c_str(), std::string(b).c_str()); \
        g_failures++; \
    } \
} while(0)

#define EXPECT_EQ_SZ(a, b) do { \
    g_tests++; \
    if ((a) != (b)) { \
        fprintf(stderr, "FAIL %s:%d: %s = %zu, expected %zu\n", \
                __FILE__, __LINE__, #a, (size_t)(a), (size_t)(b)); \
        g_failures++; \
    } \
} while(0)

#define EXPECT_TRUE(x) do { \
    g_tests++; \
    if (!(x)) { \
        fprintf(stderr, "FAIL %s:%d: expected true: %s\n", __FILE__, __LINE__, #x); \
        g_failures++; \
    } \
} while(0)

// Returns true if utf8_buf ends with a complete UTF-8 sequence.
// This is the exact logic from engine.cu generate().
static bool utf8_complete(const std::string& utf8_buf) {
    if (utf8_buf.empty()) return true;
    int pos = (int)utf8_buf.size() - 1;
    while (pos >= 0 && ((unsigned char)utf8_buf[pos] & 0xC0) == 0x80) pos--;
    if (pos >= 0) {
        unsigned char lead = utf8_buf[pos];
        int expected = 1;
        if ((lead & 0xE0) == 0xC0) expected = 2;
        else if ((lead & 0xF0) == 0xE0) expected = 3;
        else if ((lead & 0xF8) == 0xF0) expected = 4;
        int have = (int)utf8_buf.size() - pos;
        return have >= expected;
    }
    return true;
}

// Simulate the streaming token callback from generate().
// Takes a sequence of raw byte strings (one per "token decode"), feeds them
// through the UTF-8 buffering logic, and returns the list of flushed strings.
static std::vector<std::string> simulate_utf8_streaming(const std::vector<std::string>& token_bytes) {
    std::vector<std::string> flushed;
    std::string utf8_buf;

    for (const auto& decoded : token_bytes) {
        utf8_buf += decoded;
        if (utf8_complete(utf8_buf)) {
            flushed.push_back(utf8_buf);
            utf8_buf.clear();
        }
    }
    // Simulate EOS flush
    if (!utf8_buf.empty()) {
        flushed.push_back(utf8_buf);
    }
    return flushed;
}

static void test_ascii() {
    printf("  test_ascii...\n");
    auto result = simulate_utf8_streaming({"H", "ello", " world"});
    EXPECT_EQ_SZ(result.size(), (size_t)3);
    EXPECT_EQ_STR(result[0], "H");
    EXPECT_EQ_STR(result[1], "ello");
    EXPECT_EQ_STR(result[2], " world");
}

static void test_complete_2byte_in_one_token() {
    printf("  test_complete_2byte_in_one_token...\n");
    // é = 0xC3 0xA9 (2-byte UTF-8)
    auto result = simulate_utf8_streaming({"\xC3\xA9"});
    EXPECT_EQ_SZ(result.size(), (size_t)1);
    EXPECT_EQ_STR(result[0], "\xC3\xA9");
}

static void test_split_2byte_across_tokens() {
    printf("  test_split_2byte_across_tokens...\n");
    // é split across two tokens
    auto result = simulate_utf8_streaming({"\xC3", "\xA9"});
    EXPECT_EQ_SZ(result.size(), (size_t)1);
    EXPECT_EQ_STR(result[0], "\xC3\xA9");
}

static void test_emoji_4byte() {
    printf("  test_emoji_4byte...\n");
    // 😀 = F0 9F 98 80 (4-byte UTF-8)
    // Delivered as 4 separate single-byte tokens (worst case)
    auto result = simulate_utf8_streaming({"\xF0", "\x9F", "\x98", "\x80"});
    EXPECT_EQ_SZ(result.size(), (size_t)1);
    EXPECT_EQ_STR(result[0], "\xF0\x9F\x98\x80");
}

static void test_emoji_split_2_2() {
    printf("  test_emoji_split_2_2...\n");
    // 😀 split as 2+2 bytes
    auto result = simulate_utf8_streaming({"\xF0\x9F", "\x98\x80"});
    EXPECT_EQ_SZ(result.size(), (size_t)1);
    EXPECT_EQ_STR(result[0], "\xF0\x9F\x98\x80");
}

static void test_emoji_split_3_1() {
    printf("  test_emoji_split_3_1...\n");
    // 😀 split as 3+1 bytes
    auto result = simulate_utf8_streaming({"\xF0\x9F\x98", "\x80"});
    EXPECT_EQ_SZ(result.size(), (size_t)1);
    EXPECT_EQ_STR(result[0], "\xF0\x9F\x98\x80");
}

static void test_emoji_split_1_3() {
    printf("  test_emoji_split_1_3...\n");
    // 😀 split as 1+3 bytes
    auto result = simulate_utf8_streaming({"\xF0", "\x9F\x98\x80"});
    EXPECT_EQ_SZ(result.size(), (size_t)1);
    EXPECT_EQ_STR(result[0], "\xF0\x9F\x98\x80");
}

static void test_mixed_ascii_and_emoji() {
    printf("  test_mixed_ascii_and_emoji...\n");
    // "Hi 😀!" with emoji split across tokens
    auto result = simulate_utf8_streaming({"Hi ", "\xF0\x9F", "\x98\x80", "!"});
    EXPECT_EQ_SZ(result.size(), (size_t)3);
    EXPECT_EQ_STR(result[0], "Hi ");
    EXPECT_EQ_STR(result[1], "\xF0\x9F\x98\x80");
    EXPECT_EQ_STR(result[2], "!");
}

static void test_3byte_cjk() {
    printf("  test_3byte_cjk...\n");
    // 中 = E4 B8 AD (3-byte UTF-8)
    // Split as 1+1+1
    auto result = simulate_utf8_streaming({"\xE4", "\xB8", "\xAD"});
    EXPECT_EQ_SZ(result.size(), (size_t)1);
    EXPECT_EQ_STR(result[0], "\xE4\xB8\xAD");
}

static void test_3byte_split_2_1() {
    printf("  test_3byte_split_2_1...\n");
    auto result = simulate_utf8_streaming({"\xE4\xB8", "\xAD"});
    EXPECT_EQ_SZ(result.size(), (size_t)1);
    EXPECT_EQ_STR(result[0], "\xE4\xB8\xAD");
}

static void test_multiple_emoji_sequence() {
    printf("  test_multiple_emoji_sequence...\n");
    // 🎉🔥 = F0 9F 8E 89 F0 9F 94 A5
    // First emoji complete in one token, second split across two
    auto result = simulate_utf8_streaming({
        "\xF0\x9F\x8E\x89",         // 🎉 complete
        "\xF0\x9F",                  // 🔥 first 2 bytes
        "\x94\xA5"                   // 🔥 last 2 bytes
    });
    EXPECT_EQ_SZ(result.size(), (size_t)2);
    EXPECT_EQ_STR(result[0], "\xF0\x9F\x8E\x89");
    EXPECT_EQ_STR(result[1], "\xF0\x9F\x94\xA5");
}

static void test_eos_flushes_incomplete() {
    printf("  test_eos_flushes_incomplete...\n");
    // If generation stops mid-emoji (EOS), the buffer should still be flushed.
    // This simulates the EOS flush in generate().
    auto result = simulate_utf8_streaming({"\xF0\x9F"});
    // Should flush the incomplete bytes rather than dropping them
    EXPECT_EQ_SZ(result.size(), (size_t)1);
    EXPECT_EQ_STR(result[0], "\xF0\x9F");
}

static void test_empty_tokens() {
    printf("  test_empty_tokens...\n");
    auto result = simulate_utf8_streaming({"", "Hi", "", "!"});
    EXPECT_EQ_SZ(result.size(), (size_t)4);
    EXPECT_EQ_STR(result[1], "Hi");
    EXPECT_EQ_STR(result[3], "!");
}

static void test_ascii_after_incomplete_emoji() {
    printf("  test_ascii_after_incomplete_emoji...\n");
    // Token 1: start of 4-byte emoji
    // Token 2: remaining 3 bytes of emoji + some ASCII
    auto result = simulate_utf8_streaming({"\xF0", "\x9F\x98\x80 hi"});
    EXPECT_EQ_SZ(result.size(), (size_t)1);
    EXPECT_EQ_STR(result[0], "\xF0\x9F\x98\x80 hi");
}

int main() {
    printf("=== UTF-8 Streaming Buffer Tests ===\n");

    test_ascii();
    test_complete_2byte_in_one_token();
    test_split_2byte_across_tokens();
    test_emoji_4byte();
    test_emoji_split_2_2();
    test_emoji_split_3_1();
    test_emoji_split_1_3();
    test_mixed_ascii_and_emoji();
    test_3byte_cjk();
    test_3byte_split_2_1();
    test_multiple_emoji_sequence();
    test_eos_flushes_incomplete();
    test_empty_tokens();
    test_ascii_after_incomplete_emoji();

    printf("\n%d/%d tests passed\n", g_tests - g_failures, g_tests);
    return g_failures > 0 ? 1 : 0;
}
