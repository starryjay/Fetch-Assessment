"""
Task 3: Training Considerations
Discuss the implications and advantages of each scenario and explain your rationale as to 
how the model should be trained given the following:

    1. If the entire network should be frozen.
            This would be applicable if the task we are using the network for is very similar 
        to the pre-trained task and the model has already learned the necessary features.
        In that case, the weights learned by the original model would be sufficient for 
        the new task. This would be zero-shot inference, so the outputs may not be accurate.

    2. If only the transformer backbone should be frozen.
            This would allow for the task-specific heads to be trained while keeping the
        original weights of the transformer backbone intact. This is useful when the new task
        is somewhat similar to the pre-trained task, but the task-specific heads need to be
        fine-tuned to adapt to the new task. However, this would not be as useful as a fully
        fine-tuned model, since the sentence embeddings would not be updated. Performance
        would be better than the first scenario, though.

    3. If only one of the task-specific heads (either for Task A or Task B) should be frozen.
            This would allow for the other task-specific head to be trained while the frozen head
        would not be updated. This is useful when one of the tasks is more similar to the
        pre-trained task than the other, and we want to keep the weights of the more similar
        task intact without overwriting them with the other task's weights - i.e. asymmetric 
        fine-tuning that prevents catastrophic forgetting.

Consider a scenario where transfer learning can be beneficial. Explain how you would approach 
the transfer learning process, including:

    1. The choice of a pre-trained model.
            I would choose a pre-trained model that is similar to at least one of the tasks I 
        am working on. For example, if I am working on a sentiment analysis task, I would 
        choose a pre-trained model that has been trained on sentiment analysis data, and 
        prioritize models trained on data similar to my dataset. If there are multiple
        pre-trained models available that are similar to my tasks, I would then prioritize
        well-documented models. 

    2. The layers you would freeze/unfreeze.
            I would freeze the earlier layers of the transformer backbone, since these layers
        primarily pick up on high-level features such as sentence structure and grammar;
        no fine-tuning would be needed for these layers. Unfreezing the later layers of
        the transformer backbone would allow for the model's attention to be fine-tuned
        to the specific tasks I am working on. I would also unfreeze the task-specific
        heads, since these layers are specific to the tasks I am working on and would
        need to be fine-tuned to adapt to the new tasks.
    
    3. The rationale behind these choices.
            As stated above, a pre-trained model that is similar to at least one of the tasks
        I am working on, and well-documented, would be the best choice for transfer learning.
        Freezing the earlier layers of the transformer backbone would allow for the model
        to retain the high-level features learned from the pre-trained model, while unfreezing
        the later layers and task-specific heads would allow for the model to adapt to the 
        specific tasks I am working on. 
"""