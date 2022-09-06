ARG VERSION=22.06

FROM zhangchi0104/honours-sgtm:${VERSION}-runtime

COPY . /workspace
WORKDIR /workspace
