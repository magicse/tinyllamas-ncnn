#include "net.h"

#include <string>
#include <iostream>
#include <vector>
#include <cstdio>
#include <fstream>
#include <cmath>
#include <random>
#include <cstdlib>
#include <algorithm>
#include <unordered_map>

const int vocab_size = 32000, ctx_length = 1024;
const float temp = 0.5f;

std::mt19937_64 rng(0);
std::uniform_real_distribution<float> dist(0, 1);

struct bpe
{
    int max_token_length;
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> lookup;
    std::vector<float> scores;

    void load(std::string path);
    std::vector<int> encode(std::string s);
};

void bpe::load(std::string path)
{
    vocab.resize(vocab_size);
    scores.resize(vocab_size);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) exit(1);
    fread(&max_token_length, sizeof(int), 1, f);
    std::vector<char> s(max_token_length + 1);
    for (int i = 0; i < vocab_size; i++)
    {
        fread(scores.data() + i, sizeof(float), 1, f);
        int len;
        fread(&len, sizeof(int), 1, f);
        fread(s.data(), sizeof(char) * len, 1, f);
        s[len] = 0;
        vocab[i] = s.data();
        lookup[vocab[i]] = i;
    }
    fclose(f);
}

std::vector<int> bpe::encode(std::string s)
{
    std::vector<int> tokens;
    for (size_t i = 0; i < s.length(); i++)
    {
        std::string c;
        c += s[i];
        int id = lookup[c];
        tokens.push_back(id);
    }

    while (true)
    {
        float best_score = -1e10;
        int best_index = -1, best_token = -1;

        for (size_t i = 0; i + 1 < tokens.size(); i++)
        {
            auto merged = vocab[tokens[i]] + vocab[tokens[i + 1]];
            if (lookup.count(merged) && scores[lookup[merged]] > best_score)
            {
                best_score = scores[lookup[merged]];
                best_index = i;
                best_token = lookup[merged];
            }
        }

        if (best_token == -1) break;

        tokens[best_index] = best_token;
        tokens.erase(tokens.begin() + best_index + 1);
    }
    return tokens;
}

struct tinyllama
{
    ncnn::Net net;
    float topp;
    tinyllama(std::string bin, std::string param);
    int forward(const std::vector<int>& tokens);
};

tinyllama::tinyllama(std::string bin, std::string param)
{
    if (net.load_param(param.c_str())) exit(1);
    if (net.load_model(bin.c_str())) exit(1);
}

int tinyllama::forward(const std::vector<int>& tokens)
{
    std::vector<int> input(ctx_length, 0);
    for (int i = 0; i < std::min(ctx_length, (int)tokens.size()); i++)
    {
        input[ctx_length - 1 - i] = tokens[tokens.size() - 1 - i];
    }

    ncnn::Mat in(ctx_length);
    for (int i = 0; i < ctx_length; i++) ((int*)in)[i] = input[i];
    auto ex = net.create_extractor();
    ex.input("in0", in);
    // ex.set_light_mode(false);
    ncnn::Mat out;
    ex.extract("out0", out);

    std::vector<float> logits(vocab_size);
    for (int i = 0; i < vocab_size; i++) logits[i] = out[i];

    // softmax
    auto maximum = *std::max_element(logits.begin(), logits.end());
    for (int i = 0; i < vocab_size; i++) logits[i] = expf(logits[i] - maximum);
    float tot = 0;
    for (int i = 0; i < vocab_size; i++) tot += logits[i];
    for (int i = 0; i < vocab_size; i++) logits[i] /= tot * temp;
    // std::cerr << tot << std::endl;

    // top-p sampling
    // std::vector<std::pair<int, float> > a;
    // for (int i = 0; i < vocab_size; i++) a.emplace_back(i, logits[i]);
    // std::sort(a.begin(), a.end(), [](auto a, auto b) { return a.second > b.second; });

    // float sum = 0;
    // int last = 0;
    // for (int i = 0; i < vocab_size; i++)
    // {
    //     sum += a[i].second;
    //     last = i;
    //     if (sum > topp)
    //         break;
    // }

    // float r = dist(rng) * sum;
    // sum = 0;
    // for (int i = 0; i <= last; i++) {
    //     sum += a[i].second;
    //     if (sum > r) return a[i].first;
    // }

    // return a[last].first;

    return std::max_element(logits.begin(), logits.end()) - logits.begin();
}

// ./tinyllamas PROMPT OUT-TOKEN-COUNT
int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " PROMPT OUT-TOKEN-COUNT" << std::endl;
        return 1;
    }

    std::string model_bin = "model.ncnn.bin", model_param = "model.ncnn.param", tokenizer_path = "tokenizer.bin", prompt = argv[1];
    int token_count = std::stoi(argv[2]);

    tinyllama model(model_bin, model_param);
    model.topp = 0.1f;

    // tokenize prompt
    bpe tokenizer;
    tokenizer.load(tokenizer_path);

    auto tokens = tokenizer.encode(prompt);

    for (auto token : tokens) std::cout << tokenizer.vocab[token] << std::flush;

    // feed forward
    for (int _ = 0; _ < token_count; _++)
    {
        auto next = model.forward(tokens);
        std::cout << tokenizer.vocab[next] << std::flush;
        tokens.push_back(next);
    }
    std::cout << "\n";
}
