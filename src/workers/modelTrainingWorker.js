import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';
let  _globalCtx = {};
let _model = null;
const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1    
}

// normalizer os dados para o intervalo [0, 1]
const normalize = (value, min, max) => (value - min) / ((max - min) || 1);

function makeContex(products,users) {
    const ages = users.map(u => u.age);
    const prices = products.map(c => c.price);

    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);

    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    const colors = [...new Set(products.map(c => c.color))]
    const categories = [...new Set(products.map(c => c.category))]

    const colorsIndex = Object.fromEntries(
        colors.map((color,index) => {
            return [color,index];
        })
    );
    
    const categoriesIndex = Object.fromEntries(
        categories.map((categorie,index) => {
            return [categorie,index];
        })
    );

    //média de idade dos compradores
    const midAge = (minAge + maxAge) / 2;

    //soma das idades dos compradores
    const ageSums = {};
    const ageCounts = {};
    users.forEach(u => {
        u.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + u.age;
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1;
        });
    });

    const productAvgAgeNorm = Object.fromEntries(
        products.map(c => {
            const avgAge = ageCounts[c.name] ? ageSums[c.name] / ageCounts[c.name] : midAge;
            return [c.name, normalize(avgAge, minAge, maxAge)];
        })  
    )

    return {
        products,
        users,
        colorsIndex,
        categoriesIndex,
        productAvgAgeNorm,  
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,

        //price + age + color + category
        dimentions: 2 + colors.length + categories.length,
    }
     
}

const oneHotWeighted = (index,lenght,weight) => 
    tf.oneHot(index, lenght).cast('float32').mul(weight);

function encodeProduct(product, context) {   
    // normalizendo dados para ficar de 0 a 1 e aplicar o peso na recomendação

    const price = tf.tensor1d([
        normalize(product.price, context.minPrice, context.maxPrice) * WEIGHTS.price
    ]);


    const age = tf.tensor1d([
        (context.productAvgAgeNorm[product.name] ?? 0.5) * WEIGHTS.age
    ]);

    const category = oneHotWeighted(context.categoriesIndex[product.category], context.numCategories, WEIGHTS.category);
    const color = oneHotWeighted(context.colorsIndex[product.color], context.numColors, WEIGHTS.color);

    return tf.concat1d([price, age, category, color]);
}

function encodeUser(user, context) {
    if (user.purchases.length) {
        return tf.stack(
            user.purchases.map(
                product => encodeProduct(product,context)
            )
        )
        .mean(0)
        .reshape([1, context.dimentions]);
    }

    // se o usuário não tiver compras, retorna um vetor neutro (todos os valores iguais)
    return tf.concat1d([
        tf.zeros([1]), // preço zerado
        tf.tensor1d([normalize(user.age, context.minAge, context.maxAge) * WEIGHTS.age]), // idade normalizeda e ponderada
        tf.zeros([context.numCategories]), // categorias zeradas
        tf.zeros([context.numColors]) // cores zeradas  
    ]).reshape([1, context.dimentions]);

}

function createTrainingData(context) {
    const inputs = [];
    const labels = [];

    context.users
    .filter(u => u.purchases.length > 0) // filtra somente usuários que não fizeram compras
    .forEach(user => {
        const userVector = encodeUser(user, context).dataSync();
        context.products.forEach(product => {
            const productVector = encodeProduct(product, context).dataSync();
             //vetor de entrada combinando userVector e productVector
             // e um rótulo (label) indicando se o usuário comprou o produto ou não
             const label = user.purchases.some(p => p.name === product.name) ? 1 : 0;
            inputs.push([...userVector, ...productVector]);
            labels.push(label);
        });
    });

    return {
        xs: tf.tensor2d(inputs), 
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimentions: context.dimentions * 2 // userVector + productVector
    }   
}

async function configureNeuralNetAndTrain(trainData) {
  const model = tf.sequential();

  // camadas densas (fully connected) com funções de ativação ReLU 
  // é feito várias camadas para o modelo aprender representações mais complexas dos dados
  // começa maior e vai diminuindo para forçar o modelo a aprender as características mais importantes
  model.add(tf.layers.dense({ inputShape: [trainData.inputDimentions], units: 128, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));  

  // camada de saída para classificação binária
  //exemplo: 0,9 significa que o modelo acha que tem 90% de chance do usuário comprar o produto, enquanto 0,1 significa 10% de chance.
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' })); 
  
  model.compile({
    optimizer: tf.train.adam(0.01), // taxa de aprendizado
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });
  
   await model.fit(trainData.xs, trainData.ys, {
    epochs: 100,
    batchSize: 32,
    shuffle: true,
    callbacks: {
        onEpochEnd: (epoch, logs) => {
             postMessage({
                type: workerEvents.trainingLog,
                epoch: epoch,
                loss: logs.loss,
                accuracy: logs.acc
         });
        }
    }
    });

    return model;
}

async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });
    const products = await (await fetch('/data/products.json')).json();
    const context = makeContex(products,users);

    context.productVectors = products.map(product => {
        return {
            name: product.name,
            meta: {...product},
            vector: encodeProduct(product, context).dataSync(),
        }
    });

    _globalCtx = context;

    const trainData = createTrainingData(context);
    _model = await configureNeuralNetAndTrain(trainData);

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });

}

function recommend(user, context) {
    //evento quando o usuário clica para receber recomendações, o modelo precisa pegar o vetor do usuário e comparar com os vetores dos produtos para prever a probabilidade de compra de cada produto. Depois, ele pode ordenar os produtos por essa probabilidade e recomendar os mais prováveis.
    
    if (!_model) return;

    const userTensor = encodeUser(user, context).dataSync();

    // combinando o vetor do usuário com os vetores dos produtos para criar os dados de entrada para o modelo
    // para fins de estudos os vetores estão em memoria no productVectors, mas idealmente os vetores deveriam ser gerados sob demanda ou armazenados em um banco de dados para evitar o consumo excessivo de memória
    const input = context.productVectors.map(({vector}) => {
        return [...userTensor, ...vector];
    });

    const inputTensor = tf.tensor2d(input);
    const scores = _model.predict(inputTensor).dataSync();

    const recommendations = context.productVectors.map((product, index) => {
        return {
            ...product.meta,    
            name: product.name,
            score: scores[index]
        };
    });

    const sortedRecommendations = recommendations.sort((a, b) => b.score - a.score);
    
    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedRecommendations
    });
}


const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
