def training_loop(EPOCHS, MODEL, OPTIMIZER, device, train_loader, test_loader):
    summary(MODEL)
    
    if torch.cuda.is_available() is False:
        raise Exception("GPU device not found, runtime environment should be set to GPU")
    print(f"Using GPU device: {torch.cuda.get_device_name(device)}")
    
    
    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    history = {'train_loss': [], 'train_accuracy': [], 'test_loss': [],'test_accuracy': []
               ,'time':None}

    for epoch in range(EPOCHS):
        start=time.time()
        
        temp_loss = 0
        correct=0
        graphs=0
        for step,data in enumerate(train_loader):
            data.to(device)
            OPTIMIZER.zero_grad()  # Clear gradients.
            y_out = MODEL(data) # Perform a single forward pass.
            loss = criterion(y_out, data.y)  # Compute the loss solely based on the training nodes.
            loss.backward()  # Derive gradients.
            OPTIMIZER.step()  # Update parameters based on gradients.
            
            temp_loss=temp_loss+loss.detach().item()
            pred = y_out.argmax(dim=1)  # Use the class with highest probability.
            correct =correct + int((pred == data.y).sum())
            graphs = graphs + data.num_graphs
            
        train_acc=float("{:.2f}".format(correct / graphs))
        train_loss=float("{:.2f}".format(temp_loss/(step+1)))
        
         
        temp_loss = 0
        correct=0
        graphs=0  
        for step,data in enumerate(test_loader):
            data.to(device)
            y_out =MODEL(data)
            loss = criterion(y_out,  data.y) 
            
            temp_loss =  temp_loss + loss.detach().item()
            pred = y_out.argmax(dim=1)  # Use the class with highest probability.
            correct =correct + int((pred == data.y).sum())
            graphs = graphs + data.num_graphs
        
        test_acc=float("{:.2f}".format(correct / graphs))
        test_loss=float("{:.2f}".format(temp_loss/(step+1)))
        
        end=time.time()
        
        print(f"Epoch: {epoch} | Train loss: {train_loss} | Train accuracy: {train_acc}  | Test loss: {test_loss} | Test accuracy: {test_acc}| Time: {end-start}")
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['test_accuracy'].append(test_acc)
        history['test_loss'].append(test_loss)
        history['time']=end-start
    return history