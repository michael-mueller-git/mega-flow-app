# Usage

```yaml
apiVersion: helm.toolkit.fluxcd.io/v2
kind: HelmRelease
metadata:
  name: ${APP_NAME}
  namespace: ${APP_NAMESPACE}
spec:
  interval: 10m
  chart:
    spec:
      chart: app-template
      version: 4.0.1
      sourceRef:
        kind: HelmRepository
        name: bjw-s-charts
        namespace: flux-system

  values:
    defaultPodOptions:
      runtimeClassName: "nvidia"
    controllers:
      ${APP_NAME}:
        containers:
          app:
            image:
              repository: git.gpu.lan/r/mega-flow
              tag: "latest@sha256:53ad1717741a52e1d326a96923ae0bff34dcf80e618d6de09b7ebd83239a9283"
              pullPolicy: Always
            env:
              NVIDIA_VISIBLE_DEVICES: "0"
              NVIDIA_DRIVER_CAPABILITIES: all

    service:
      webui:
        controller: ${APP_NAME}
        ports:
          http:
            port: 7860

    ingress:
      webui:
        className: traefik
        annotations:
          traefik.ingress.kubernetes.io/router.entrypoints: websecure
        hosts:
          - host: &ingress1 "mega-flow.${SECRET_DOMAIN}"
            paths:
              - path: /
                pathType: Prefix
                service:
                  identifier: webui
                  port: http
        tls:
          - hosts:
              - *ingress1

```
