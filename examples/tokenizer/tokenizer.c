#include "../../src/ican.h"

int main(void) {
    Tokenizer *tokenizer = tokenizer_alloc();
    load_tokenizer_from_json(tokenizer, "tokenizer.json");
    load_tokenizer_config_from_json(tokenizer, "tokenizer-config.json");
    printf("Added Tokens = %s \n", cJSON_Print(tokenizer->added_tokens));
    printf("Vocab = %s \n", cJSON_Print(tokenizer->vocab));
    printf("BPE RULES = %s \n", cJSON_Print(tokenizer->bpe_rules));

    Iray1D *output = fits_on_text(tokenizer, "hello World <|endoftext|>");
    iray1d_print(output);

    iray1d_free(output);
    
    tokenizer_free(tokenizer);
    return 0;
}