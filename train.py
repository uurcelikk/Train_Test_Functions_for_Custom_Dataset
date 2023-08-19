import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    
    # Modeli eğitim moduna geçir
    model.train()

    # Eğitim kaybı ve doğruluğu için başlangıç değerleri
    train_loss, train_acc = 0,0

    # Dataloader üzerinde döngü 
    for batch, (X,y) in enumerate(dataloader):
        
        # Veri ve etiketleri seçilen cihaza taşı
        X,y = X.to(device), y.to(device)

        #Modelden tahminleri elde et
        y_pred = model(X)
        # Kaybı hesapla    
        loss = loss_fn(y_pred,y)
        train_loss += loss.item()

        #Optimizasyon için gradyanları sıfırla
        optimizer.zero_grad()

        #Geriye doğru hesaplama yaparak gradyanları hesapla
        loss.backward()

        #parametreleri güncelle
        optimizer.step()


        # tahminleri sınıf indekslerine dönüştür
        # Doğru tahmin sayısını hesapla ve doğruluk değerini güncelle
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim =1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        """
        üstteki kod için:
        y_pred değişkeni, model tarafından yapılan tahminlerin olasılık dağılımlarını içerir.
        Bu dağılımları sınıf indekslerine dönüştürmek için iki işlem yapıyoruz:
        1) İlk olarak, torch.softmax(y_pred, dim=1) ifadesi, modelin çıktılarını softmax fonksiyonu ile normalizasyon yaparak olasılık dağılımlarına dönüştürür.
        dim=1 parametresi, normalizasyonun her bir veri örneği için yapıldığını belirtir.
        Ardından, torch.argmax(..., dim=1) ifadesi, her veri örneği için en yüksek olasılığa sahip sınıfın indeksini seçer. 
        Bu, tahmin edilen sınıfı belirlememizi sağlar.
        2)(y_pred_class == y) ifadesi, modelin tahmin ettiği sınıf indeksleri ile gerçek sınıf indekslerini karşılaştırır 
        ve eşleşen indekslerde True döndürür.
        sum() fonksiyonu True değerlerin sayısını hesaplar ve .item() metodu ile sonucu bir sayı olarak çıkartır.

        """
    

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc