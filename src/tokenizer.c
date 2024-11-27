#ifndef TOKENIZER_H

#define TOKENIZER_H

#include "ican.h" 

Tokenizer *tokenizer_alloc() {
    Tokenizer *tokenizer = (Tokenizer*)ICAN_MALLOC(sizeof(Tokenizer));
    tokenizer->vocab = cJSON_CreateObject();
    tokenizer->vocab_size = 0;
    tokenizer->added_tokens = cJSON_CreateArray();
    tokenizer->bpe_rules = cJSON_CreateObject();
    tokenizer->bpe_rule_size = 0;
    tokenizer->unk_token = NULL;
    tokenizer->bos_token = NULL;
    tokenizer->eos_token = NULL;
    tokenizer->max_length = 0;
    return tokenizer;
}

void tokenizer_free(Tokenizer *tokenizer) {
    cJSON_Delete(tokenizer->vocab);
    cJSON_Delete(tokenizer->added_tokens);
    cJSON_Delete(tokenizer->bpe_rules);
    if(tokenizer->unk_token) {
        ICAN_FREE(tokenizer->unk_token);
    }
    if(tokenizer->bos_token) {
        ICAN_FREE(tokenizer->bos_token);
    }
    if(tokenizer->eos_token) {
        ICAN_FREE(tokenizer->eos_token);
    }
    ICAN_FREE(tokenizer);
}

void load_vocab_from_json(Tokenizer *tokenizer, const char *file_path) {
    FILE *file = fopen(file_path, "r");
    ASSERT_MSG(file != NULL, "File not found");

    fseek(file, 0, SEEK_END);
    size_t length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *data = (char *)ICAN_MALLOC(length + 1);
    fread(data, 1, length, file);
    data[length] = '\0';
    fclose(file);

    cJSON_Delete(tokenizer->vocab);
    tokenizer->vocab = cJSON_Parse(data);
    tokenizer->vocab_size = 0; /** Todo: Add vocab size */
    ICAN_FREE(data);
}

void load_bpe_from_text(Tokenizer *tokenizer, const char *file_path) {
    FILE *file = fopen(file_path, "r");
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    ASSERT_MSG(file != NULL, "File not found");

    fseek(file, 0, SEEK_END);
    size_t length = ftell(file);
    fseek(file, 0, SEEK_SET);

    while (read = getline(&line, &len, file) != -1) {

        char *first = strtok(line, " ");
        char *second = strtok(NULL, " ");

        cJSON_AddItemToObject(tokenizer->bpe_rules, first, cJSON_CreateString(second));
    }
    
    tokenizer->bpe_rule_size = length;
    ICAN_FREE(line);
    fclose(file);
}

void load_tokenizer_config_from_json(Tokenizer *tokenizer, const char *file_path) {
    FILE *file = fopen(file_path, "r");
    ASSERT_MSG(file != NULL, "File not found");

    fseek(file, 0, SEEK_END);
    size_t length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *data = (char *)ICAN_MALLOC(length + 1);
    fread(data, 1, length, file);
    data[length] = '\0';
    fclose(file);

    cJSON *json = cJSON_Parse(data);

    tokenizer->bos_token = cJSON_GetObjectItem(json, "bos_token")->valuestring;
    tokenizer->eos_token = cJSON_GetObjectItem(json, "eos_token")->valuestring;
    tokenizer->max_length = cJSON_GetObjectItem(json, "model_max_length")->valueint;

    ICAN_FREE(data);
    cJSON_Delete(json);
}

void load_tokenizer_from_json(Tokenizer *tokenizer, const char *file_path) {
    FILE *file = fopen(file_path, "r");
    ASSERT_MSG(file != NULL, "File not found");

    fseek(file, 0, SEEK_END);
    size_t length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char *data = (char *)ICAN_MALLOC(length + 1);
    fread(data, 1, length, file);
    data[length] = '\0';
    fclose(file);

    cJSON_Delete(tokenizer->vocab);
    cJSON_Delete(tokenizer->added_tokens);

    cJSON *json = cJSON_Parse(data);
    ICAN_FREE(data);

    ASSERT_MSG(json != NULL, "Json could not parsed");

    /** TODO: Add padding and truncation */

    cJSON *added_tokens = cJSON_GetObjectItem(json, "added_tokens");
    tokenizer->added_tokens = cJSON_Duplicate(added_tokens, 1);

    cJSON *model = cJSON_GetObjectItem(json, "model");

    cJSON *vocab = cJSON_GetObjectItem(model, "vocab");
    tokenizer->vocab = cJSON_Duplicate(vocab, 1);
    tokenizer->vocab_size = 0; /** Todo: Do it without traversy */

    cJSON *merges = cJSON_GetObjectItem(model, "merges");

    tokenizer->bpe_rule_size = cJSON_GetArraySize(merges);
    
    for (size_t i = 0; i < tokenizer->bpe_rule_size; i++) {
        cJSON *merge = cJSON_GetArrayItem(merges, i);
        char *pair = strdup(merge->valuestring);

        char *first = strtok(pair, " ");
        char *second = strtok(NULL, " ");

        cJSON_AddItemToObject(tokenizer->bpe_rules, first, cJSON_CreateString(second));
        ICAN_FREE(pair);
    }
    tokenizer->unk_token = cJSON_GetObjectItem(model, "unk_token")->valuestring;

    cJSON_Delete(json);
}

bool match_token_with_properties(const char *token, cJSON *properties) {
    cJSON *content = cJSON_GetObjectItem(properties, "content");
    if (!content || !cJSON_IsString(content)) {
        return false;
    }

    char *temp = strdup(token);

    if (cJSON_GetObjectItem(properties, "lstrip") && cJSON_IsTrue(cJSON_GetObjectItem(properties, "lstrip"))) {
        while (*temp == ' ') temp++;
    }

    if (cJSON_GetObjectItem(properties, "rstrip") && cJSON_IsTrue(cJSON_GetObjectItem(properties, "rstrip"))) {
        size_t len = strlen(temp);
        while (len > 0 && temp[len - 1] == ' ') {
            len--;
        }
        temp[len] = '\0';
    }

    bool match = (strcmp(temp, content->valuestring) == 0);
    ICAN_FREE(temp);
    return match;
}

Iray1D *fits_on_text(Tokenizer *tokenizer, const char *text) {
    Iray1D *output = iray1d_alloc(tokenizer->max_length);
    
    iray1d_fill(output, 0);

    char *temp = strdup(text);
    char *token = strtok(temp, " ");
    size_t index = 0;

    while (token != NULL && index < tokenizer->max_length) {
        cJSON *indexData = cJSON_GetObjectItem(tokenizer->vocab, token);
        if(!indexData) {
            cJSON *added_token = NULL;
            cJSON_ArrayForEach(added_token, tokenizer->added_tokens) {
                if(match_token_with_properties(token, added_token)) {
                    indexData = cJSON_GetObjectItem(added_token, "id");
                    break;
                }
            };
        }
        if(!indexData) {
            indexData = cJSON_GetObjectItem(tokenizer->vocab, tokenizer->unk_token);
        }
        if(indexData) {
            output->data[index] = indexData->valueint;
        }

        index++;
        token = strtok(NULL, " ");
    }
    ICAN_FREE(temp);
    return output;
}

#endif // !TOKENIZER_H