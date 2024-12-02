#include "data_structure/linked_list.h"

SingleLinkedList *single_linked_list_alloc(int data) {
    SingleLinkedList *node = (SingleLinkedList *) ICAN_MALLOC(sizeof(SingleLinkedList));
    ASSERT(node != NULL);
    node->data = data;
    node->next = (SingleLinkedList *)NULL;
    return node;
}

void single_linked_list_free(SingleLinkedList *head) {
    SingleLinkedList *current = head;
    SingleLinkedList *next;
    while (current != NULL) {
        next = current->next;
        ICAN_FREE(current);
        current = next;
    }
}

SingleLinkedList *single_linked_list_insert_beginning(SingleLinkedList *head, int data) {
    SingleLinkedList *node = single_linked_list_alloc(data);
    node->next = head;
    return node;
}

SingleLinkedList *single_linked_list_insert_end(SingleLinkedList *head, int data) {
    SingleLinkedList *node = single_linked_list_alloc(data);
    SingleLinkedList *end;
    for (end = head; end->next != NULL; end = end->next);
    end->next = node;
    return head;
}

SingleLinkedList *single_linked_list_insert_pos(SingleLinkedList *head, int data, int16 pos) {
    if (pos == 1) return single_linked_list_insert_beginning(head, data);
    SingleLinkedList *search;
    int16 count;
    for (count = 1, search = head; count < pos - 1; search = search->next, count++);
    if (search == NULL) return head;
    SingleLinkedList *node = single_linked_list_alloc(data);
    node->next = search->next;
    search->next = node;
    return head;
}

SingleLinkedList *single_linked_list_delete_beginning(SingleLinkedList *head) {
    SingleLinkedList *temp = head;
    head = head->next;
    temp->next = NULL;
    single_linked_list_free(temp);
    return head;
}

SingleLinkedList *single_linked_list_delete_end(SingleLinkedList *head) {
    SingleLinkedList *end;
    SingleLinkedList *prev;
    for (end = head, prev = NULL; end->next != NULL; prev = end, end = end->next);
    prev->next = NULL;
    single_linked_list_free(end);
    return head;
}

SingleLinkedList *single_linked_list_delete_pos(SingleLinkedList *head, int16 pos) {
    if (pos == 1) return single_linked_list_delete_beginning(head);
    SingleLinkedList *search;
    int16 count;
    for (count = 1, search = head; count < pos - 1; search = search->next, count++);
    if (search == NULL || search->next == NULL) return head;
    SingleLinkedList *temp = search->next;
    search->next = search->next->next;
    temp->next = NULL;
    single_linked_list_free(temp);
    return head;
}

SingleLinkedList *single_linked_list_search(SingleLinkedList *head, int target) {
    SingleLinkedList *current = head;
    while (current != NULL) {
        if (current->data == target) {
            return current;
        }
        current = current->next;
    }
    return current;
}

void single_linked_list_traverse(SingleLinkedList *head) {
    SingleLinkedList *current = head;
    while (current != NULL) {
        printf("%d ", current->data);
        if(current->next != NULL) {
            printf("-> ");
        }
        current = current->next;
    }
}

DoubleLinkedList *double_linked_list_alloc(int data) {
    DoubleLinkedList *node = (DoubleLinkedList *) ICAN_MALLOC(sizeof(DoubleLinkedList));
    ASSERT(node != NULL);
    node->data = data;
    node->next = (DoubleLinkedList *)NULL;
    node->prev = (DoubleLinkedList *)NULL;
    return node;   
}

void double_linked_list_free(DoubleLinkedList *head) {
    DoubleLinkedList *current = head;
    DoubleLinkedList *next;
    while (current != NULL) {
        next = current->next;
        ICAN_FREE(current);
        current = next;
    }
}

DoubleLinkedList *double_linked_list_insert_beginning(DoubleLinkedList *head, int data) {
    DoubleLinkedList *node = double_linked_list_alloc(data);
    node->next = head;
    head->prev = node;
    return node;
}

DoubleLinkedList *double_linked_list_insert_end(DoubleLinkedList *head, int data) {
    DoubleLinkedList *node = double_linked_list_alloc(data);
    DoubleLinkedList *end;
    for (end = head; end->next != NULL; end = end->next);
    end->next = node;
    node->prev = end;
    return head;
}

DoubleLinkedList *double_linked_list_insert_pos(DoubleLinkedList *head, int data, int16 pos) {
    if (pos == 1) return double_linked_list_insert_beginning(head, data);
    DoubleLinkedList *search;
    int16 count;
    for (count = 1, search = head; count < pos - 1; search = search->next, count++);
    if (search == NULL) return head;
    DoubleLinkedList *node = double_linked_list_alloc(data);
    node->next = search->next;
    search->next = node;
    node->prev = search;
    search->next->prev = node;
    return head;
}

DoubleLinkedList *double_linked_list_search(DoubleLinkedList *head, int target) {
    DoubleLinkedList *current = head;
    while (current != NULL) {
        if (current->data == target) {
            return current;
        }
        current = current->next;
    }
    return current;
}

void double_linked_list_traverse_forward(DoubleLinkedList *head) {
    DoubleLinkedList *current = head;
    while (current != NULL) {
        printf("%d ", current->data);
        if(current->next != NULL) {
            printf(" <-> ");
        }
        current = current->next;
    }
}

void double_linked_list_traverse_backward(DoubleLinkedList *head) {
    DoubleLinkedList *current = head;
    while (current != NULL) {
        printf("%d ", current->data);
        if(current->prev != NULL) {
            printf(" <-> ");
        }
        current = current->prev;
    }
}