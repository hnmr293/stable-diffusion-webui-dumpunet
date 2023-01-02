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

        const update_info = () => {
            const layer = app.querySelector('#dumpunet-layer select').value;
            const info = JSON.parse(app.querySelector('#dumpunet-layer_setting').textContent)[layer];
            const
                w = +width_slider.value,
                h = +height_slider.value,
                iw = Math.max(1, Math.ceil(w / 64)),
                ih = Math.max(1, Math.ceil(h / 64));
            info[0][1] *= ih;
            info[0][2] *= iw;
            info[1][1] *= ih;
            info[1][2] *= iw;
            app.querySelector('#dumpunet-layerinfo').innerHTML = `
[Layer Info]<br/>
Name:&nbsp;&nbsp;&nbsp;${layer}<br/>
Input:&nbsp;&nbsp;(${info[0].join(',')})<br/>
Outout:&nbsp;(${info[1].join(',')})<br/>
`.trim();
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
