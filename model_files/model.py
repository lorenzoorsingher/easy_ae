import torch.nn as nn


class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()

        #([(W - K + 2P)/S] + 1)

        #input img [BS, 1, 250,250]

        # out [BS, 16, 42, 42]
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=3,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )

        # out [BS, 32, 7, 7]
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=16,              
                out_channels=32,            
                kernel_size=5,              
                stride=3,                   
                padding=2,                  
            ),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        
        #output_size = strides * (input_size-1) + kernel_size - 2*padding

        # out [BS, 16, 19, 19]
        self.trans1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=5,
                stride=3,
                padding=2
            ),
            nn.ReLU()
        )

        # out [BS, 8, 55, 55]
        self.trans2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=8,
                kernel_size=5,
                stride=3,
                padding=2
            ),
            nn.ReLU()
        )

        # out [BS, 3, 163, 163]
        self.trans3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=3,
                kernel_size=5,
                stride=3,
                padding=2
            ),
            nn.ReLU()
        )

        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.trans1(x) 
        x = self.trans2(x)
        x = self.trans3(x)   
        output = self.sig(x)
        return output, x    # return x for visualization


class SAE2(nn.Module):
    def __init__(self):
        super(SAE2, self).__init__()

        #([(W - K + 2P)/S] + 1)

        #input img [BS, 1, 250,250]

        # out [BS, 8, 82, 82]
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=8,            
                kernel_size=5,              
                stride=3,                   
                padding=0,                  
            ),                              
            nn.ReLU(),                      
        )

        # out [BS, 16, 26, 26]
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=8,              
                out_channels=16,            
                kernel_size=5,              
                stride=3,                   
                padding=0                  
            ),     
            nn.ReLU(),                      
        )
        
        # out [BS, 32, 8, 8]
        self.conv3 = nn.Sequential(         
            nn.Conv2d(
                in_channels=16,              
                out_channels=32,            
                kernel_size=5,              
                stride=3,                   
                padding=0                  
            ),     
            nn.ReLU(),                      
        )
        #output_size = strides * (input_size-1) + kernel_size - 2*padding

        self.flat = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(2048,1024),
                    nn.ReLU(),
                    nn.Linear(1024,2048),
                    nn.ReLU()
        )
        


        # out [BS, 16, 26, 26]
        self.trans1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=5,
                stride=3,
                padding=0
            ),
            nn.ReLU()
        )

        # out [BS, 8, 80, 80]
        self.trans2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=8,
                kernel_size=5,
                stride=3,
                padding=0
            ),
            nn.ReLU()
        )

        # out [BS, 3, 242, 242]
        self.trans3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=3,
                kernel_size=5,
                stride=3,
                padding=0
            ),
            nn.ReLU()
        )

        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = x.view(x.size(0), 16, 26, 26)
        x = self.trans1(x) 
        x = self.trans2(x)
        output = self.trans3(x)   
        #output = self.sig(x)
        return output, x    # return x for visualization



class SimplerAE(nn.Module):
    def __init__(self):
        super(SimplerAE, self).__init__()

        #([(W - K + 2P)/S] + 1)

        #input img [BS, 1, 80,80]

        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=8,            
                kernel_size=5,              
                stride=2,                   
                padding=0,                  
            ),                              
            nn.ReLU(),                      
        )# out [BS, 8, 38, 38]

        # in [BS, 8, 38, 38]
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=8,              
                out_channels=16,            
                kernel_size=5,              
                stride=3,                   
                padding=0                  
            ),     
            nn.ReLU(),                      
        )# out [BS, 16, 12, 12]
        
        # in [BS, 16, 12, 12]
        self.flat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2304, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2304),
            nn.ReLU()
        )# out [BS, 2304]

        #output_size = strides * (input_size-1) + kernel_size - 2*padding
        
        # in [BS, 16, 12, 12]
        self.trans1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=8,
                kernel_size=5,
                stride=3,
                padding=0
            ),
            nn.ReLU()
        )# out [BS, 8, 38, 38]

        # in [BS, 8, 38, 38]
        self.trans2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=3,
                kernel_size=5,
                stride=2,
                padding=0
            ),
            nn.ReLU()
        )# out [BS, 3, 79, 79]

        #self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flat(x)
        x = x.view(x.size(0), 16, 12, 12)
        x = self.trans1(x) 
        x = self.trans2(x)
        output = self.tanh(x)
         
        #output = self.sig(x)
        return output, x    # return x for visualization




class SimplerAE2(nn.Module):
    def __init__(self):
        super(SimplerAE2, self).__init__()

        #([(W - K + 2P)/S] + 1)

        #input img [BS, 1, 80,80]

        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=8,            
                kernel_size=5,              
                stride=2,                   
                padding=0,                  
            ),                              
            nn.ReLU(),
            nn.BatchNorm2d(8)                      
        )# out [BS, 8, 38, 38]

        # in [BS, 8, 38, 38]
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=8,              
                out_channels=16,            
                kernel_size=5,              
                stride=3,                   
                padding=0                  
            ),     
            nn.ReLU(),
            nn.BatchNorm2d(16)                        
        )# out [BS, 16, 12, 12]
        
        # in [BS, 16, 12, 12]
        self.flat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Linear(512, 2304),
            nn.ReLU()
        )# out [BS, 2304]

        #output_size = strides * (input_size-1) + kernel_size - 2*padding
        
        # in [BS, 16, 12, 12]
        self.trans1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=8,
                kernel_size=5,
                stride=3,
                padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm2d(8)  
        )# out [BS, 8, 38, 38]

        # in [BS, 8, 38, 38]
        self.trans2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=3,
                kernel_size=5,
                stride=2,
                padding=0
            ),
            nn.ReLU(),
            nn.BatchNorm2d(3)  
        )# out [BS, 3, 79, 79]

        #self.sig = nn.Sigmoid()
        #self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flat(x)
        x = x.view(x.size(0), 16, 12, 12)
        x = self.trans1(x) 
        output = self.trans2(x)
        #output = self.tanh(x)
         
        #output = self.sig(x)
        return output, x    # return x for visualization
