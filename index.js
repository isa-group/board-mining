const https = require('https');
const fs = require('fs');
var boardinfo = "";

function iterate(){

    const data = fs.readFileSync('file.txt', 'UTF-8');
    const lines = data.split(/\r?\n/);
     

    try {
        lines.forEach((line) => {
            https.get({
                hostname: 'trello.com',
                path: `/b/`+line+`.json`,
                headers: {'User-Agent': `${Math.random().toString(16).substring(2,16)}`}
            }, (r) => {
                var data = "";
                r.on('data', (d) => {
                    data+=d;
                })
                r.on('close', () => {
                    boardinfo = JSON.parse(data);
                });
            })
            
            var actions = [];
            
            (function untilDeath(beforeval) {
            https.get({
                hostname: 'api.trello.com',
                path: `/1/boards/`+line+`/actions?limit=1000${beforeval ? `&before=${beforeval}` : ``}`,
                headers: {'User-Agent': `${Math.random().toString(16).substring(2,16)}`}
            }, (r) => {
                var cmpdta = "";
                r.on('data', (d) => {
                    cmpdta+=d;
                    
                })
                r.on('close', () => {
                    cmpdta = JSON.parse(cmpdta);
                    console.log('date %s, size %d', beforeval, cmpdta.length);
                    console.log(cmpdta.length)
                    if(cmpdta.length < 1000) {
                        if(cmpdta.length) actions.push(cmpdta);
                        return makeFile({}, [].concat.apply([], actions));
                    } else
                    var lastKey = Object.keys(cmpdta).sort().reverse()[0];
                    //console.log(cmpdta[lastKey])
                    untilDeath(cmpdta[lastKey]["date"]);
                    cmpdta.pop();
                    actions.push(cmpdta);
                });
            
                r.on('error', () => {
                    throw new Error('-----HTTPS Error Occurred, Please retry :(');
                });
            });
            })();

        }); 
    } catch (error) {
        return true;
      }
}


function makeFile(trelloBoard, actions) {
    trelloBoard["actions"] = actions;
    let vari = trelloBoard["actions"][0]["id"];
    fs.createWriteStream(vari+'.json');
    fs.writeFile(`${vari}.json`, JSON.stringify(trelloBoard, null, `\t`), (c) => {
        if(c) console.log(c);
    });
}

iterate()