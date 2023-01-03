onUiUpdate(() => {
    if (globalThis.DumpUnet) return;
    const DumpUnet = {};
    globalThis.DumpUnet = DumpUnet;

    DumpUnet.applySizeCallback = function () {
        if (globalThis.DumpUnet.applySizeCallbackCalled) return;

        const app = gradioApp();
        if (!app || app === document) return;

        const labels = Array.of(...app.querySelectorAll('#tab_txt2img label'));
        const width_label = labels.find(x => x.textContent.trim() === "Width");
        const height_label = labels.find(x => x.textContent.trim() === "Height");
        const steps_label = labels.find(x => x.textContent.trim() === "Sampling Steps");
        if (!width_label || !height_label || !steps_label) return;

        const width_slider = app.querySelector(`#${width_label.htmlFor}`);
        const height_slider = app.querySelector(`#${height_label.htmlFor}`);
        const steps_slider = app.querySelector(`#${steps_label.htmlFor}`)
        if (!width_slider || !height_slider || !steps_slider) return;
        //if (+width_slider.dataset.dumpunetHooked && +height_slider.dataset.dumpunetHooked) return
        //
        //const value_hook = ele => {
        //    const proto = Object.getPrototypeOf(ele);
        //    const old_desc = Object.getOwnPropertyDescriptor(proto, 'value');
        //    Object.defineProperty(ele, 'value', {
        //        get: function () { return old_desc.get.apply(this, arguments); },
        //        set: function () {
        //            const old_value = this.value;
        //            old_desc.set.apply(this, arguments);
        //            const new_value = this.value;
        //            const ev = new CustomEvent('imagesizesliderchange', { detail: { old_value: old_value }, bubbles: true });
        //            ele.dispatchEvent(ev);
        //        }
        //    });
        //    ele.dataset.dumpunetHooked = 1;
        //};
        //
        //value_hook(width_slider);
        //value_hook(height_slider);

        globalThis.DumpUnet.applySizeCallbackCalled = true;

        const layer_names = [
            // 'IN@@',
            'IN00', 'IN01', 'IN02', 'IN03', 'IN04', 'IN05', 'IN06', 'IN07', 'IN08', 'IN09', 'IN10', 'IN11',
            'M00',
            'OUT00', 'OUT01', 'OUT02', 'OUT03', 'OUT04', 'OUT05', 'OUT06', 'OUT07', 'OUT08', 'OUT09', 'OUT10', 'OUT11',
            // 'OUT$$',
        ];

        const parse_layers_token = token => {
            const layer1 = String.raw`((?:IN|OUT)\d\d|M00)`;
            const re_layer = new RegExp(String.raw`^\s*${layer1}\s*$`)
            const re_range = new RegExp(String.raw`^\s*${layer1}\s*-\s*${layer1}\s*(?:\(\s*\+?\s*(\d+)\s*\))?\s*$`)

            const m1 = re_layer.exec(token);
            const m2 = re_range.exec(token);
            if (m1 !== null) {
                const idx = layer_names.indexOf(m1[1]);
                return [idx];
            }
            if (m2 !== null) {
                let [_, from, to, step] = m2;
                from = layer_names.indexOf(from);
                to = layer_names.indexOf(to);
                step = step ? Math.max(+step, 1) : 1;
                if (from < 0 || to < 0) {
                    return [];
                }
                const result = [];
                for (let n = from; n <= to; n += step) {
                    result.push(n);
                }
                return result;
            }
            return [];
        };

        const parse_layers = input => {
            if (input === undefined || input === null || input === '') {
                return [];
            }

            const layers = input.split(',').flatMap(t => parse_layers_token(t));
            return [...new Set(layers)].sort().map(n => layer_names[n]);
        };

        const layer_input_ele = app.querySelector('#dumpunet-layer textarea');
        layer_input_ele.addEventListener('input', e => {
            const input = layer_input_ele.value;
            const layers = parse_layers(input);
            layer_input_ele.style.backgroundColor = layers.length == 0 ? 'pink' : 'white';
        }, false);

        const update_info = () => {
            const layer_input = layer_input_ele.value;
            const layers = parse_layers(layer_input);

            const
                w = +width_slider.value,
                h = +height_slider.value,
                iw = Math.max(1, Math.ceil(w / 64)),
                ih = Math.max(1, Math.ceil(h / 64));

            let html = '[Selected Layer Info]<br/>';

            for (let layer of layers) {
                const info = JSON.parse(app.querySelector('#dumpunet-layer_setting').textContent)[layer];
                info[0][1] *= ih;
                info[0][2] *= iw;
                info[1][1] *= ih;
                info[1][2] *= iw;
                html += `
Name:&nbsp;&nbsp;&nbsp;${layer}<br/>
Input:&nbsp;&nbsp;(${info[0].join(',')})<br/>
Outout:&nbsp;(${info[1].join(',')})<br/>
---<br/>
`.trim();
            }

            app.querySelector('#dumpunet-layerinfo').innerHTML = html.trim();
        };

        //app.addEventListener('imagesizesliderchange', e => {
        //    //console.log(e.detail.old_value, e.target.value);
        //    update_info();
        //}, false);

        app.addEventListener('input', update_info, false);
        app.addEventListener('change', update_info, false);

        update_info();
    };

    onUiUpdate(DumpUnet.applySizeCallback);
});
